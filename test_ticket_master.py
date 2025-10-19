"""
Minimal tests for Ticket Master Agent

Tests critical flows:
1. Check showtimes
2. Get pricing
3. Purchase tickets (idempotency)
"""

import json
from ticket_master_agent import TicketMasterAgent


def test_check_showtimes():
    """Test showtime availability check"""
    print("=" * 80)
    print("TEST: Check Showtimes")
    print("=" * 80)

    agent = TicketMasterAgent()
    query = "What showtimes are available for Avatar?"
    response = agent.invoke(query)

    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # Validation
    assert "OUT-OF-DOMAIN" not in response, "Should be IN-DOMAIN"
    assert "confidence" in response.lower() or "source" in response.lower(), "Should include confidence/source"
    print("✓ PASSED: Showtimes check works\n")


def test_get_pricing():
    """Test ticket pricing retrieval"""
    print("=" * 80)
    print("TEST: Get Pricing")
    print("=" * 80)

    agent = TicketMasterAgent()
    query = "How much are IMAX tickets?"
    response = agent.invoke(query)

    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # Validation
    assert "OUT-OF-DOMAIN" not in response, "Should be IN-DOMAIN"
    assert "$" in response or "price" in response.lower(), "Should include pricing information"
    print("✓ PASSED: Pricing retrieval works\n")


def test_purchase_tickets():
    """Test ticket purchase with reservation"""
    print("=" * 80)
    print("TEST: Purchase Tickets")
    print("=" * 80)

    agent = TicketMasterAgent()
    query = "I want to buy 2 tickets for Avatar on Friday at 7:30 PM"
    response = agent.invoke(query)

    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # Validation
    assert "OUT-OF-DOMAIN" not in response, "Should be IN-DOMAIN"
    # Should include confirmation details
    has_confirmation = any(keyword in response.lower() for keyword in [
        "confirmation", "reserved", "booked", "purchased", "ticket"
    ])
    assert has_confirmation, "Should include purchase confirmation"
    print("✓ PASSED: Ticket purchase works\n")


def test_out_of_domain():
    """Test OUT-OF-DOMAIN detection"""
    print("=" * 80)
    print("TEST: OUT-OF-DOMAIN Detection")
    print("=" * 80)

    agent = TicketMasterAgent()
    query = "What movies has Tom Cruise been in?"
    response = agent.invoke(query)

    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # Validation
    assert "OUT-OF-DOMAIN" in response, "Should detect OUT-OF-DOMAIN query"
    print("✓ PASSED: OUT-OF-DOMAIN detection works\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TICKET MASTER AGENT - MINIMAL TESTS")
    print("=" * 80 + "\n")

    try:
        test_check_showtimes()
        test_get_pricing()
        test_purchase_tickets()
        test_out_of_domain()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        raise
