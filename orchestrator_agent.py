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

# Import specialist agents
from movie_specialist_agent import invoke_movie_specialist
from ticket_master_agent import invoke_ticket_master
from vendor_agent import invoke_vendor

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


# Supervisor routing function
def create_supervisor_chain():
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

Query reformulation:
- Use specific movie names from previous responses
- For FINISH, set empty string

Select from: {options}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Who should act next? Select from: {options}")
    ]).partial(options=str(options), members=", ".join(members))

    # Use structured output to get routing decision with query reformulation
    supervisor = prompt | llm.with_structured_output(RouteDecision)
    # print(f"--- supervisor: {supervisor}")
    return supervisor


# Agent node functions
def movie_specialist_node(state: OrchestratorState) -> Dict[str, Any]:
    """Execute movie specialist agent"""
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
    # Invoke the movie specialist
    response = invoke_movie_specialist(query, state.get("session_state"))

    result_message = AIMessage(
        content=f"[Movie Specialist]: {response}",
        name="movie_specialist"
    )

    return {
        "messages": [result_message],
    }


def ticket_master_node(state: OrchestratorState) -> Dict[str, Any]:
    """Execute ticket master agent"""
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
    # Invoke the ticket master
    response = invoke_ticket_master(query, state.get("session_state"))

    result_message = AIMessage(
        content=f"[Ticket Master]: {response}",
        name="ticket_master"
    )

    return {
        "messages": [result_message],
    }


def vendor_node(state: OrchestratorState) -> Dict[str, Any]:
    """Execute vendor agent"""
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
    # Invoke the vendor
    response = invoke_vendor(query, state.get("session_state"))

    result_message = AIMessage(
        content=f"[Vendor]: {response}",
        name="vendor"
    )

    return {
        "messages": [result_message],
    }


def supervisor_node(state: OrchestratorState) -> Dict[str, Any]:
    """Supervisor decides which agent to route to and reformulates the query"""
    supervisor_chain = create_supervisor_chain()

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
        }

    # Safety check: ensure decision is valid
    if not hasattr(decision, 'next'):
        print(f"[ERROR] Invalid decision structure: {decision}")
        return {
            "next": "FINISH",
            "reformulated_query": "",
        }

    # Extract routing decision
    next_agent = decision.next
    reformulated_query = decision.reformulated_query if hasattr(decision, 'reformulated_query') else ""

    # Update state with next agent and reformulated query
    return {
        "next": next_agent,
        "reformulated_query": reformulated_query,
    }


# Build the orchestration graph
def create_orchestrator_graph():
    """Create the multi-agent orchestration graph"""

    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("movie_specialist", movie_specialist_node)
    workflow.add_node("ticket_master", ticket_master_node)
    workflow.add_node("vendor", vendor_node)

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


# Main orchestrator interface
class OrchestratorAgent:
    """Orchestrator Agent - coordinates specialist agents"""

    def __init__(self):
        """Initialize the orchestrator with the graph"""
        self.graph = create_orchestrator_graph()
        self.session_state = {}

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

        # Prepare initial state
        initial_state: OrchestratorState = {
            "messages": [HumanMessage(content=query)],
            "next": "",
            "session_state": self.session_state,
            "reformulated_query": ""
        }

        # Execute the graph
        result = self.graph.invoke(initial_state)

        # Extract all agent responses
        responses = []
        for message in result["messages"]:
            if isinstance(message, AIMessage) and message.content:
                responses.append(message.content)


        # Return combined response or last response
        if responses:
            # If multiple responses from different agents, combine them
            if len(responses) > 1:
                return "\n\n".join(responses)
            else:
                return responses[-1]
        else:
            return "I'm sorry, I couldn't process your request. Please try again."

    def reset_session(self):
        """Reset session state"""
        self.session_state = {}


# Global instance (singleton pattern)
_orchestrator_instance = None


def get_orchestrator() -> OrchestratorAgent:
    """Get or create the global Orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorAgent()
    return _orchestrator_instance


def invoke_orchestrator(query: str, session_state: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to invoke the Orchestrator

    Args:
        query: User's question or request
        session_state: Optional session state for context

    Returns:
        str: Agent's response content
    """
    orchestrator = get_orchestrator()
    return orchestrator.invoke(query, session_state)


if __name__ == "__main__":
    print("Testing Orchestrator Agent...\n")

    # Test 1: Movie query
    print("=" * 80)
    print("Test 1: Movie Query")
    print("=" * 80)
    query1 = "I want to watch a sci-fi movie with great visual effects"
    response1 = invoke_orchestrator(query1)
    print(f"---Query: {query1}")
    print(f"Response: {response1}\n")

    # Test 2: Ticket query
    print("=" * 80)
    print("Test 2: Ticket Query")
    print("=" * 80)
    query2 = "What are the showtimes for Avatar on Friday?"
    response2 = invoke_orchestrator(query2)
    print(f"---Query: {query2}")
    print(f"Response: {response2}\n")

    # Test 3: Vendor query
    print("=" * 80)
    print("Test 3: Vendor Query")
    print("=" * 80)
    query3 = "How much is a large popcorn and soda?"
    response3 = invoke_orchestrator(query3)
    print(f"---Query: {query3}")
    print(f"Response: {response3}\n")


    # Test 4: Multi-domain query
    print("=" * 80)
    print("Test 4: Multi-Domain Query")
    print("=" * 80)
    query4 = "I want to watch Avatar, get 2 tickets for 7pm, and order popcorn"
    response4 = invoke_orchestrator(query4)
    print(f"---Query: {query4}")
    print(f"Response: {response4}\n")

    # Test 5: Multi-domain query
    print("=" * 80)
    print("Test 5: Multi-Domain Query")
    print("=" * 80)
    query5 = "Can I get two IMAX tickets for Friday 19:30 and tell me the plot about it"
    response5 = invoke_orchestrator(query5)
    print(f"---Query: {query5}")
    print(f"Response: {response5}\n")

    # Test 6: Multi-domain query
    print("=" * 80)
    print("Test 6: Multi-Domain Query")
    print("=" * 80)
    query6 = "Can I get two IMAX tickets for Friday 19:30 and add a large caramel popcorn? Also, is the new sci‑fi any good?"
    response6 = invoke_orchestrator(query6)
    print(f"Query: {query6}")
    print(f"Response: {response6}\n")