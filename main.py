"""
Main entry point for the Multi-Agent Movie Theater System

Allows users to interact directly with specialist agents or the orchestrator.
If a specialist agent returns OUT-OF-DOMAIN, automatically routes to orchestrator.

Agent instances are maintained across conversations to preserve message history.
"""

from movie_specialist_agent import MovieSpecialistAgent
from ticket_master_agent import TicketMasterAgent
from vendor_agent import VendorAgent
from orchestrator_agent import OrchestratorAgent


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 80)
    print("🎬 MULTI-AGENT MOVIE THEATER SYSTEM 🍿")
    print("=" * 80)
    print("\nWelcome! You can talk directly to specialist agents or let the orchestrator")
    print("route your query automatically.\n")


def print_menu():
    """Print agent selection menu"""
    print("\nAvailable Agents:")
    print("  1. Movie Specialist - Recommendations, plots, actor info")
    print("  2. Ticket Master - Showtimes, pricing, reservations")
    print("  3. Vendor - Snack pricing, inventory, orders")
    print("  4. Orchestrator - Auto-routing for multi-domain queries")
    print("  5. Exit")
    print()


def handle_agent_response(response: str, query: str, session_state: dict, orchestrator) -> str:
    """
    Handle agent response. If OUT-OF-DOMAIN detected, route to orchestrator.

    Args:
        response: Agent's response
        query: Original user query
        session_state: Current session state
        orchestrator: Orchestrator agent instance

    Returns:
        Final response (either from agent or orchestrator)
    """
    if "OUT-OF-DOMAIN" in response:
        print("\n⚠️  Agent cannot handle this request. Escalating to Orchestrator...\n")
        return orchestrator.invoke(query, session_state)
    return response


def main():
    """Main interactive loop"""
    print_banner()

    # Initialize agent instances - maintains message history
    movie_specialist = MovieSpecialistAgent()
    ticket_master = TicketMasterAgent()
    vendor = VendorAgent()
    orchestrator = OrchestratorAgent()

    # Session state to maintain context across conversations
    session_state = {}

    print("ℹ️  Agent instances initialized. Message history will be maintained.\n")

    while True:
        print_menu()
        choice = input("Select an agent (1-5): ").strip()

        if choice == "5":
            print("\n👋 Thank you for using the Movie Theater System. Enjoy your movie!\n")
            break

        if choice not in ["1", "2", "3", "4"]:
            print("\n❌ Invalid choice. Please select 1-5.\n")
            continue

        # Get user query
        print("\n" + "-" * 80)
        query = input("Your query: ").strip()

        if not query:
            print("\n❌ Empty query. Please try again.\n")
            continue

        print("-" * 80 + "\n")

        # Route to selected agent
        try:
            if choice == "1":
                print("🎬 [Movie Specialist]")
                print("⏳ Processing your request...\n")
                response = movie_specialist.invoke(query, session_state)
                final_response = handle_agent_response(response, query, session_state, orchestrator)

            elif choice == "2":
                print("🎟️  [Ticket Master]")
                print("⏳ Processing your request...\n")
                response = ticket_master.invoke(query, session_state)
                final_response = handle_agent_response(response, query, session_state, orchestrator)

            elif choice == "3":
                print("🍿 [Vendor]")
                print("⏳ Processing your request...\n")
                response = vendor.invoke(query, session_state)
                final_response = handle_agent_response(response, query, session_state, orchestrator)

            else:  # choice == "4"
                print("🎭 [Orchestrator]")
                print("⏳ Processing your request...\n")
                final_response = orchestrator.invoke(query, session_state)

            # Display response
            print(final_response)
            print("\n" + "=" * 80)

        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            print("Please try again or select a different agent.\n")


if __name__ == "__main__":
    main()
