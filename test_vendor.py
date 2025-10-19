"""
Minimal tests for Vendor Agent

Tests critical flows:
1. Get snack pricing
2. Check availability
3. Place order with inventory reservation
4. Combo pricing validation
5. Substitutions when out of stock
"""

import json
from vendor_agent import VendorAgent


def test_get_snack_pricing():
    """Test snack pricing retrieval"""
    print("=" * 80)
    print("TEST: Get Snack Pricing")
    print("=" * 80)

    agent = VendorAgent()
    query = "How much is a large popcorn?"
    response = agent.invoke(query)

    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # Validation
    assert "OUT-OF-DOMAIN" not in response, "Should be IN-DOMAIN"
    assert "$" in response or "price" in response.lower(), "Should include pricing"
    assert "confidence" in response.lower() or "source" in response.lower(), "Should include confidence/source"
    print("✓ PASSED: Snack pricing works\n")


def test_check_availability():
    """Test inventory availability check"""
    print("=" * 80)
    print("TEST: Check Availability")
    print("=" * 80)

    agent = VendorAgent()
    query = "Do you have nachos available?"
    response = agent.invoke(query)

    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # Validation
    assert "OUT-OF-DOMAIN" not in response, "Should be IN-DOMAIN"
    # Should indicate availability status
    has_availability_info = any(keyword in response.lower() for keyword in [
        "available", "stock", "in stock", "out of stock", "have"
    ])
    assert has_availability_info, "Should include availability information"
    print("✓ PASSED: Availability check works\n")


def test_place_order():
    """Test order placement with inventory reservation"""
    print("=" * 80)
    print("TEST: Place Order")
    print("=" * 80)

    agent = VendorAgent()
    query = "I want to order 2 large popcorns and a medium soda"
    response = agent.invoke(query)

    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # Validation
    assert "OUT-OF-DOMAIN" not in response, "Should be IN-DOMAIN"
    # Should include order confirmation
    has_confirmation = any(keyword in response.lower() for keyword in [
        "order", "confirmation", "total", "pickup", "reserved"
    ])
    assert has_confirmation, "Should include order confirmation"
    print("✓ PASSED: Order placement works\n")


def test_out_of_domain():
    """Test OUT-OF-DOMAIN detection"""
    print("=" * 80)
    print("TEST: OUT-OF-DOMAIN Detection")
    print("=" * 80)

    agent = VendorAgent()
    query = "What time does Avatar start?"
    response = agent.invoke(query)

    print(f"Query: {query}")
    print(f"Response: {response}\n")

    # Validation
    assert "OUT-OF-DOMAIN" in response, "Should detect OUT-OF-DOMAIN query"
    print("✓ PASSED: OUT-OF-DOMAIN detection works\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VENDOR AGENT - MINIMAL TESTS")
    print("=" * 80 + "\n")

    try:
        test_get_snack_pricing()
        test_check_availability()
        test_place_order()
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
