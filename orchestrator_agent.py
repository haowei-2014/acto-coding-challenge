"""
Orchestrator Agent

Routes user queries to appropriate specialist agents (Movie, Ticket, Vendor).
Acts as theater manager coordinating multi-domain requests.
"""

import json
import os
from typing import Annotated, Dict, Any, Optional, List, Literal, TypedDict
from dotenv import load_dotenv
import operator
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import specialist agents
from movie_specialist_agent import MovieSpecialistAgent
from ticket_master_agent import TicketMasterAgent
from vendor_agent import VendorAgent

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Define the state structure with reducer for messages
class OrchestratorState(TypedDict):
    """State shared across the orchestration graph"""
    messages: Annotated[List[BaseMessage], operator.add]  # Use operator.add to append messages
    next: str  # Which agent to route to next
    session_state: Dict[str, Any]  # Shared session context
    reformulated_query: str  # Reformulated query for the next specialist
    final_answer: str  # Final answer when routing to FINISH


# Define routing options
members = ["movie_specialist", "ticket_master", "vendor", "FINISH"]
options = members


# Define routing decision structure using Pydantic
class RouteDecision(BaseModel):
    """Decision on which agent to route to"""
    next: Literal["movie_specialist", "ticket_master", "vendor", "FINISH"] = Field(
        description="The next agent to route to, or FINISH if complete"
    )
    reasoning: str = Field(description="Reasoning for the routing decision")
    reformulated_query: str = Field(
        description="Query reformulated for the specific specialist. Empty string for FINISH."
    )
    final_answer: str = Field(
        default="",
        description="If next is FINISH, provide the final answer to the user based on conversation history. Otherwise empty string."
    )


# Main orchestrator interface
class OrchestratorAgent:
    """Orchestrator Agent - coordinates specialist agents"""

    def __init__(self):
        """Initialize the orchestrator with persistent specialist agents and graph"""
        # Create persistent specialist agent instances
        self.movie_specialist = MovieSpecialistAgent()
        self.ticket_master = TicketMasterAgent()
        self.vendor = VendorAgent()

        # Create the orchestration graph
        self.graph = self._create_orchestrator_graph()
        self.session_state = {}
        self.message_history = []  # Track conversation history across invocations

    def _create_supervisor_chain(self):
        """Create the supervisor that routes to specialist agents"""

        system_prompt = """You are the Orchestrator at a movie theater.

Specialist agents:
1. movie_specialist: Recommendations, details, plots, actor info
2. ticket_master: Showtimes, pricing, reservations, purchases
3. vendor: Snack pricing, inventory, orders

Routing rules:
- Do NOT over-route
- Only route when user EXPLICITLY requests that service
- Stop routing when all user requests are fulfilled

Examples:
- "I want to watch a comedy" → movie_specialist (implicit recommendation request)
- "Recommend a comedy" → movie_specialist (explicit request)
- "Book 3 tickets for Inception" → ticket_master only (don't also route to movie_specialist)
- "Reserve seats for Dune and buy candy" → ticket_master, then vendor
- "Tell me about Dune and get me tickets" → movie_specialist, then ticket_master
- "Get me tickets and tell me the plot" → ticket_master first (to get movie name), then movie_specialist
- "Can I have nachos?" → vendor only

Routing order:
- If user asks for tickets AND movie info without specifying movie name: ticket_master first, then movie_specialist
- Get concrete information (movie names, bookings) before abstract information (plots, reviews)

After an agent responds:
- Look at original user message and list ALL explicit requests
- Check which requests have been handled (agent names in history)
- Only route if there are EXPLICIT unhandled requests remaining
- Each request should be handled by exactly one agent
- When all explicit requests done, route to FINISH

Special cases - route to FINISH immediately:
- Questions about conversation history that can be answered from context
- Follow-up questions that don't require new information from specialists
- When routing to FINISH, provide a final_answer based on the conversation history

Query reformulation:
- Use specific movie names from previous responses
- For FINISH, set empty string but provide final_answer

Select from: {options}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Who should act next? Select from: {options}")
        ]).partial(options=str(options), members=", ".join(members))

        # Use structured output to get routing decision with query reformulation
        supervisor = prompt | llm.with_structured_output(RouteDecision)
        return supervisor

    def _movie_specialist_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Execute movie specialist agent with error handling"""
        # Use reformulated query from supervisor, or fall back to original message
        query = state.get("reformulated_query", "")

        if not query:
            # Fallback: find the original user message
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break

        messages = state.get("messages")
        print(f"--- movie node: {query}, messages: {messages}")

        try:
            # Invoke the movie specialist (already has retry logic)
            response = self.movie_specialist.invoke(query, state.get("session_state"))
            result_message = AIMessage(
                content=f"[Movie Specialist]: {response}",
                name="movie_specialist"
            )
        except Exception as e:
            # Partial result: specialist failed but orchestrator continues
            error_msg = f"Movie Specialist encountered an error: {str(e)[:100]}"
            result_message = AIMessage(
                content=f"[Movie Specialist]: ⚠️ {error_msg}",
                name="movie_specialist"
            )

        return {
            "messages": [result_message],
        }

    def _ticket_master_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Execute ticket master agent with error handling"""
        # Use reformulated query from supervisor, or fall back to original message
        query = state.get("reformulated_query", "")

        if not query:
            # Fallback: find the original user message
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break

        messages = state.get("messages")
        print(f"--- ticket node: {query}, messages: {messages}")

        try:
            # Invoke the ticket master (already has retry logic)
            response = self.ticket_master.invoke(query, state.get("session_state"))
            result_message = AIMessage(
                content=f"[Ticket Master]: {response}",
                name="ticket_master"
            )
        except Exception as e:
            # Partial result: specialist failed but orchestrator continues
            error_msg = f"Ticket Master encountered an error: {str(e)[:100]}"
            result_message = AIMessage(
                content=f"[Ticket Master]: ⚠️ {error_msg}",
                name="ticket_master"
            )

        return {
            "messages": [result_message],
        }

    def _vendor_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Execute vendor agent with error handling"""
        # Use reformulated query from supervisor, or fall back to original message
        query = state.get("reformulated_query", "")

        if not query:
            # Fallback: find the original user message
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break

        messages = state.get("messages")
        print(f"--- vendor node: {query}, messages: {messages}")

        try:
            # Invoke the vendor (already has retry logic)
            response = self.vendor.invoke(query, state.get("session_state"))
            result_message = AIMessage(
                content=f"[Vendor]: {response}",
                name="vendor"
            )
        except Exception as e:
            # Partial result: specialist failed but orchestrator continues
            error_msg = f"Vendor encountered an error: {str(e)[:100]}"
            result_message = AIMessage(
                content=f"[Vendor]: ⚠️ {error_msg}",
                name="vendor"
            )

        return {
            "messages": [result_message],
        }

    def _supervisor_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Supervisor decides which agent to route to and reformulates the query"""
        supervisor_chain = self._create_supervisor_chain()

        # Use messages from state
        messages = state["messages"].copy()

        # Get routing decision with reformulated query
        try:
            decision = supervisor_chain.invoke({"messages": messages})
        except Exception as e:
            print(f"[ERROR] Supervisor decision failed: {e}")
            # Fallback to FINISH if decision fails
            return {
                "next": "FINISH",
                "reformulated_query": "",
                "final_answer": "I apologize, I encountered an error processing your request."
            }

        # Safety check: ensure decision is valid
        if not hasattr(decision, 'next'):
            print(f"[ERROR] Invalid decision structure: {decision}")
            return {
                "next": "FINISH",
                "reformulated_query": "",
                "final_answer": "I apologize, I encountered an error processing your request."
            }

        # Extract routing decision
        next_agent = decision.next
        reformulated_query = decision.reformulated_query if hasattr(decision, 'reformulated_query') else ""
        final_answer = decision.final_answer if hasattr(decision, 'final_answer') else ""

        # Update state with next agent, reformulated query, and final answer
        return {
            "next": next_agent,
            "reformulated_query": reformulated_query,
            "final_answer": final_answer
        }

    def _create_orchestrator_graph(self):
        """Create the multi-agent orchestration graph"""

        workflow = StateGraph(OrchestratorState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("movie_specialist", self._movie_specialist_node)
        workflow.add_node("ticket_master", self._ticket_master_node)
        workflow.add_node("vendor", self._vendor_node)

        # Add edges from specialist agents back to supervisor
        workflow.add_edge("movie_specialist", "supervisor")
        workflow.add_edge("ticket_master", "supervisor")
        workflow.add_edge("vendor", "supervisor")

        # Conditional routing from supervisor
        def should_continue(state: OrchestratorState) -> str:
            """Determine next node based on supervisor decision"""
            next_agent = state.get("next", "FINISH")

            if next_agent == "FINISH":
                return "end"
            else:
                return next_agent

        workflow.add_conditional_edges(
            "supervisor",
            should_continue,
            {
                "movie_specialist": "movie_specialist",
                "ticket_master": "ticket_master",
                "vendor": "vendor",
                "end": END
            }
        )

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Compile the graph
        graph = workflow.compile()

        return graph

    def invoke(self, query: str, session_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Invoke the orchestrator with a user query

        Args:
            query: User's question or request
            session_state: Optional session state for context

        Returns:
            str: Final response from the orchestration
        """
        # Use provided session state or default
        if session_state:
            self.session_state.update(session_state)

        # Add new user message to history
        self.message_history.append(HumanMessage(content=query))

        # Prepare initial state with full conversation history
        initial_state: OrchestratorState = {
            "messages": self.message_history.copy(),
            "next": "",
            "session_state": self.session_state,
            "reformulated_query": "",
            "final_answer": ""
        }

        # Track messages before invocation
        messages_before_count = len(self.message_history)

        # Execute the graph
        result = self.graph.invoke(initial_state)

        # Update message history with all messages from result
        self.message_history = result["messages"]

        # Check if there's a final_answer (supervisor routed to FINISH immediately)
        if result.get("final_answer"):
            # Add final answer to message history
            final_answer_msg = AIMessage(content=result["final_answer"])
            self.message_history.append(final_answer_msg)
            # result["messages"].append(final_answer_msg)
            # return result["final_answer"]

        # Extract only NEW agent responses (added during this invocation)
        new_responses = []
        for message in result["messages"][messages_before_count:]:
            if isinstance(message, AIMessage) and message.content:
                new_responses.append(message.content)

        # Return combined response or last response
        if new_responses:
            # If multiple responses from different agents, combine them
            if len(new_responses) > 1:
                return "\n\n".join(new_responses)
            else:
                return new_responses[-1]
        else:
            return "I'm sorry, I couldn't process your request. Please try again."

    def reset_session(self):
        """Reset session state and message history"""
        self.session_state = {}
        self.message_history = []


if __name__ == "__main__":
    print("Testing Orchestrator Agent...\n")

    # Create a single orchestrator instance for all tests
    orchestrator = OrchestratorAgent()

    query5 = "I want to watch Avatar, get 2 tickets for 7pm, and order popcorn"
    response5 = orchestrator.invoke(query5)
    print(response5)
    print("---------")

    query5 = "What snack did you buy?"
    response5 = orchestrator.invoke(query5)
    print(response5)

    exit()



    # Test 1: Movie query
    print("=" * 80)
    print("Test 1: Movie Query")
    print("=" * 80)
    query1 = "I want to watch a sci-fi movie with great visual effects"
    response1 = orchestrator.invoke(query1)
    print(f"---Query: {query1}")
    print(f"Response: {response1}\n")

    # Test 2: Ticket query
    print("=" * 80)
    print("Test 2: Ticket Query")
    print("=" * 80)
    query2 = "What are the showtimes for Avatar on Friday?"
    orchestrator = OrchestratorAgent()
    response2 = orchestrator.invoke(query2)
    print(f"---Query: {query2}")
    print(f"Response: {response2}\n")

    # Test 3: Vendor query
    print("=" * 80)
    print("Test 3: Vendor Query")
    print("=" * 80)
    query3 = "How much is a large popcorn and soda?"
    orchestrator = OrchestratorAgent()
    response3 = orchestrator.invoke(query3)
    print(f"---Query: {query3}")
    print(f"Response: {response3}\n")


    # Test 4: Multi-domain query
    print("=" * 80)
    print("Test 4: Multi-Domain Query")
    print("=" * 80)
    query4 = "I want to watch Avatar, get 2 tickets for 7pm, and order popcorn"
    orchestrator = OrchestratorAgent()
    response4 = orchestrator.invoke(query4)
    print(f"---Query: {query4}")
    print(f"Response: {response4}\n")

    # Test 5: Multi-domain query
    print("=" * 80)
    print("Test 5: Multi-Domain Query")
    print("=" * 80)
    query5 = "Can I get two IMAX tickets for Friday 19:30 and tell me the plot about it"
    orchestrator = OrchestratorAgent()
    response5 = orchestrator.invoke(query5)
    print(f"---Query: {query5}")
    print(f"Response: {response5}\n")

    # Test 6: Multi-domain query
    print("=" * 80)
    print("Test 6: Multi-Domain Query")
    print("=" * 80)
    query6 = "Can I get two IMAX tickets for Friday 19:30 and add a large caramel popcorn? Also, is the new sci‑fi any good?"
    orchestrator = OrchestratorAgent()
    response6 = orchestrator.invoke(query6)
    print(f"Query: {query6}")
    print(f"Response: {response6}\n")

    # Test 7: Complex cross-domain - recommendations, tickets, and snacks
    print("=" * 80)
    print("Test 7: Complex Cross-Domain Query")
    print("=" * 80)
    query7 = "Suggest a thriller for tonight, book 3 tickets for the 8pm showing, and get me nachos and two sodas"
    orchestrator = OrchestratorAgent()
    response7 = orchestrator.invoke(query7)
    print(f"Query: {query7}")
    print(f"Response: {response7}\n")

    # Test 8: Nuanced cross-domain - tickets with plot request for different movie
    print("=" * 80)
    print("Test 8: Nuanced Cross-Domain Query")
    print("=" * 80)
    query8 = "Reserve 4 seats for The Matrix on Saturday afternoon and also tell me what Interstellar is about"
    response8 = orchestrator.invoke(query8)
    print(f"Query: {query8}")
    print(f"Response: {response8}\n")

    # Test 9: Ambiguous cross-domain - implicit ticket request with food order
    print("=" * 80)
    print("Test 9: Ambiguous Cross-Domain Query")
    print("=" * 80)
    query9 = "I'm planning to see the new Batman movie with my family of 4, can you check if there's an evening show and also order a combo deal?"
    orchestrator = OrchestratorAgent()
    response9 = orchestrator.invoke(query9)
    print(f"Query: {query9}")
    print(f"Response: {response9}\n")