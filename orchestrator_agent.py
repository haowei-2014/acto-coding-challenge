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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Define the state structure with reducer for messages
class OrchestratorState(TypedDict):
    """State shared across the orchestration graph"""
    messages: Annotated[List[BaseMessage], operator.add]  # Use operator.add to append messages
    next: str  # Which agent to route to next
    session_state: Dict[str, Any]  # Shared session context
    agents_called: List[str]  # Track which agents have been called
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

    system_prompt = """You are the Orchestrator at a movie theater, managing three specialist agents.

Your specialist agents:
1. movie_specialist: Handles movie recommendations, details, plot summaries, actor information
2. ticket_master: Handles showtimes, pricing, ticket purchases, reservations
3. vendor: Handles snack pricing, inventory, food/beverage orders

Your responsibilities:
- Analyze user queries and route to the appropriate specialist agent
- Handle multi-domain queries by routing to multiple agents in sequence
- Maintain session context (movie selection, party size, cart items)
- Coordinate cross-domain requests efficiently
- ONLY route to specialists when their specific expertise is needed
- When task is complete, route to FINISH

CRITICAL: Do not over-route! Only call specialists when the user actually asks for their domain:
- Just because a movie name is mentioned doesn't mean you need movie_specialist

Routing guidelines:
- Analyze the ORIGINAL user query to identify ALL domains mentioned:
  * Movies/actors/recommendations/plot → movie_specialist domain
  * Tickets/showtimes/pricing/reservations → ticket_master domain
  * Snacks/food/beverages/popcorn/soda/orders → vendor domain

- CRITICAL: Consider dependencies between domains before routing:
  * If user asks vague movie questions AND wants tickets:
    - Route to ticket_master FIRST to identify which specific movie they're booking
    - Then route to movie_specialist with the specific movie name from the ticket response
  * If user mentions a specific movie name AND wants tickets:
    - Can route to movie_specialist first, then ticket_master
  * General principle: Get concrete information (ticket bookings, specific items) before abstract queries (opinions, recommendations)
  * This ensures you have context to provide better answers

- If NO specialist has responded yet:
  * Determine the optimal routing order considering dependencies
  * Route to the specialist that provides concrete context first when queries are vague

- If specialists have ALREADY responded (check message history):
  * Use information from previous responses to enhance subsequent queries
  * Extract movie names, dates, or other context from previous responses
  * Reformulate queries with this concrete context
  * If ALL domains addressed → route to FINISH with empty reformulated_query

- Query Reformulation Rules:
  * Extract ONLY information relevant to the target specialist's domain
  * Remove mentions of other domains to avoid OUT-OF-DOMAIN responses
  * For vendor: focus only on snack items, prices, availability - do NOT mention movies or tickets
  * For ticket_master: MUST include movie title if mentioned in original query or previous responses
    - If user mentioned a specific movie, include it in the reformulated query
    - Example: "Get 2 IMAX tickets for Friday 19:30" should be "Get 2 IMAX tickets for Avatar on Friday at 19:30"
  * For movie_specialist: focus on movie titles, actors, recommendations
  * Make the reformulated query clear and actionable with all necessary context
  * For FINISH, set reformulated_query to empty string

- CRITICAL: Each specialist should only be called ONCE per query. Never route to the same specialist twice.

Important:
- Look at the conversation history to see which specialists have already responded
- Do not send the same query to the same specialist multiple times
- When all parts of the user's request are addressed, ALWAYS route to FINISH

Given the conversation above, who should act next?
Select one of: {options}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Select one of: {options}")
    ]).partial(options=str(options), members=", ".join(members))

    # Use structured output to get routing decision with query reformulation
    supervisor = prompt | llm.with_structured_output(RouteDecision)
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

    print(f"movie node: {query}")
    # Invoke the movie specialist
    response = invoke_movie_specialist(query, state.get("session_state"))

    result_message = AIMessage(
        content=f"[Movie Specialist]: {response}",
        name="movie_specialist"
    )

    # Track that this agent was called
    agents_called = state.get("agents_called", [])
    agents_called.append("movie_specialist")

    return {
        "messages": [result_message],
        "agents_called": agents_called,
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

    print(f"ticket node: {query}")
    # Invoke the ticket master
    response = invoke_ticket_master(query, state.get("session_state"))

    result_message = AIMessage(
        content=f"[Ticket Master]: {response}",
        name="ticket_master"
    )

    # Track that this agent was called
    agents_called = state.get("agents_called", [])
    agents_called.append("ticket_master")

    return {
        "messages": [result_message],
        "agents_called": agents_called,
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

    print(f"vendor node: {query}")
    # Invoke the vendor
    response = invoke_vendor(query, state.get("session_state"))

    result_message = AIMessage(
        content=f"[Vendor]: {response}",
        name="vendor"
    )

    # Track that this agent was called
    agents_called = state.get("agents_called", [])
    agents_called.append("vendor")

    return {
        "messages": [result_message],
        "agents_called": agents_called,
    }


def supervisor_node(state: OrchestratorState) -> Dict[str, Any]:
    """Supervisor decides which agent to route to and reformulates the query"""
    supervisor_chain = create_supervisor_chain()

    # Check which agents have already been called
    agents_called = state.get("agents_called", [])

    # If any agent has already responded, add a system message to help supervisor
    messages = state["messages"].copy()
    if agents_called:
        system_hint = SystemMessage(
            content=f"Note: The following agents have already responded to this query: {', '.join(agents_called)}. "
                    f"Either route to a different domain if needed, or route to FINISH if the query is fully satisfied. "
                    f"Remember to reformulate the query appropriately for the next specialist."
        )
        messages.append(system_hint)

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

    # Safety check: prevent routing to the same agent twice
    next_agent = decision.next
    reformulated_query = decision.reformulated_query if hasattr(decision, 'reformulated_query') else ""

    if next_agent in agents_called:
        # Force FINISH if trying to route to same agent
        next_agent = "FINISH"
        reformulated_query = ""


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
            "agents_called": [],
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
    print(f"Query: {query1}")
    print(f"Response: {response1}\n")

    # Test 2: Ticket query
    print("=" * 80)
    print("Test 2: Ticket Query")
    print("=" * 80)
    query2 = "What are the showtimes for Avatar on Friday?"
    response2 = invoke_orchestrator(query2)
    print(f"Query: {query2}")
    print(f"Response: {response2}\n")

    # Test 3: Vendor query
    print("=" * 80)
    print("Test 3: Vendor Query")
    print("=" * 80)
    query3 = "How much is a large popcorn and soda?"
    response3 = invoke_orchestrator(query3)
    print(f"Query: {query3}")
    print(f"Response: {response3}\n")


    # Test 4: Multi-domain query
    print("=" * 80)
    print("Test 4: Multi-Domain Query")
    print("=" * 80)
    query4 = "I want to watch Avatar, get 2 tickets for 7pm, and order popcorn"
    response4 = invoke_orchestrator(query4)
    print(f"Query: {query4}")
    print(f"Response: {response4}\n")

    # Test 5: Multi-domain query
    print("=" * 80)
    print("Test 5: Multi-Domain Query")
    print("=" * 80)
    query5 = "Can I get two IMAX tickets for Friday 19:30 and add a large caramel popcorn? Also, is the new sci‑fi any good?"
    response5 = invoke_orchestrator(query5)
    print(f"Query: {query5}")
    print(f"Response: {response5}\n")