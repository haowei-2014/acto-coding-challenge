"""
Vendor / Store Manager Agent

Handles snack pricing, inventory, and orders.
Max 3 tools enforced.
"""

import json
import os
from typing import Annotated, List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)


class VendorAgent:
    """Vendor/Store Manager Agent - handles snacks, inventory, and orders"""

    def __init__(self):
        """Initialize the Vendor with inventory database"""
        self.inventory_db = {}
        self.combos = {}
        self.orders_db = {}
        self._load_inventory()
        self.agent = self.create_agent()  # Create agent once, reuse for conversation history
        self.message_history = []  # Track conversation history

    def _load_inventory(self):
        """Load inventory from JSON file"""
        try:
            with open('data/vendor.json', 'r') as f:
                data = json.load(f)
                self.inventory_db = data['inventory']
                self.combos = data['combos']
            print(f"Loaded inventory with {len(self.inventory_db)} categories and {len(self.combos)} combos")
        except Exception as e:
            print(f"Error loading inventory: {e}")
            # Minimal fallback
            self.inventory_db = {
                "Popcorn": {"Large": {"price": 9.00, "stock": 40}},
                "Soda": {"Large": {"price": 7.00, "stock": 50}}
            }
            self.combos = {}

    def _find_substitutions(self, category: str, item_name: str, quantity: int):
        """
        Find alternative items in the same category when an item is out of stock.

        Args:
            category: The category of the out-of-stock item
            item_name: The name of the out-of-stock item
            quantity: The requested quantity

        Returns:
            List of substitution suggestions with availability and price
        """
        substitutions = []

        if category not in self.inventory_db:
            return substitutions

        for other_item_name, details in self.inventory_db[category].items():
            # Skip the same item
            if other_item_name == item_name:
                continue

            # Check if the alternative has sufficient stock
            if details['stock'] >= quantity:
                substitutions.append({
                    "item": f"{category} {other_item_name}",
                    "available_quantity": details['stock'],
                    "price": f"${details['price']:.2f}",
                    "reason": f"Alternative {other_item_name} size with sufficient stock"
                })

        # Limit to 2 suggestions
        return substitutions[:2]

    def _fuzzy_match_item(self, query: str):
        """
        Fuzzy match an item query to inventory items.
        Extracts key words and matches against category and item names.
        Handles synonyms like "coke" → "soda", "hotdog" → "hot dog"

        Returns: (category, item_name, details) or None if no match
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Synonym mapping (whole words only)
        synonyms = {
            'coke': 'soda',
            'pop': 'soda',
            'drink': 'soda',
            'beverage': 'soda',
            'hotdog': 'hot dog',
            'hot-dog': 'hot dog',
        }

        # Replace synonyms in query (whole words only)
        for synonym, replacement in synonyms.items():
            if synonym in query_words:
                query_lower = query_lower.replace(synonym, replacement)
                query_words = set(query_lower.split())

        # Extract size keywords
        sizes = {'small', 'medium', 'large', 'regular', 'loaded'}
        size_match = None
        for size in sizes:
            if size in query_words:
                size_match = size
                break

        # Try exact match first
        for category, items_dict in self.inventory_db.items():
            for item_name, details in items_dict.items():
                full_name = f"{category} {item_name}".lower()
                if query_lower == full_name or query_lower in full_name:
                    return (category, item_name, details)

        # Try fuzzy match - check if category name is in query
        for category, items_dict in self.inventory_db.items():
            category_lower = category.lower()

            # Check if category is mentioned in query (handle multi-word categories)
            category_words = set(category_lower.split())
            if category_lower in query_lower or any(word in query_words for word in category_words):
                # If size is specified, try to match it
                if size_match:
                    for item_name, details in items_dict.items():
                        item_lower = item_name.lower()
                        if size_match in item_lower:
                            return (category, item_name, details)

                    # If exact size not found, suggest largest available
                    if 'Large' in items_dict:
                        return (category, 'Large', items_dict['Large'])

                # Return first item in category if no size specified
                first_item = list(items_dict.keys())[0]
                return (category, first_item, items_dict[first_item])

        return None

    def get_snack_pricing(self, item_name: Optional[str] = None) -> str:
        """
        Get pricing information for snacks and combos.
        Returns prices and available items.
        """
        try:
            result = {"items": [], "combos": [], "confidence": 1.0, "source": "Theater Vendor System"}

            # Search for specific item
            if item_name:
                found = False
                search_lower = item_name.lower()

                for category, items in self.inventory_db.items():
                    for item, details in items.items():
                        # Create full name for matching (e.g., "Popcorn Large")
                        full_name = f"{category} {item}".lower()

                        # Match if search term is in full name, item name, or category
                        if (search_lower in full_name or
                            search_lower in item.lower() or
                            search_lower in category.lower()):
                            result["items"].append({
                                "category": category,
                                "name": item,
                                "price": f"${details['price']:.2f}",
                                "in_stock": details['stock'] > 0
                            })
                            found = True

                if not found:
                    return json.dumps({
                        "error": f"Item '{item_name}' not found",
                        "available_categories": list(self.inventory_db.keys()),
                        "confidence": 0.0,
                        "source": "Theater Vendor System"
                    })
            else:
                # Return all items
                for category, items in self.inventory_db.items():
                    for item, details in items.items():
                        result["items"].append({
                            "category": category,
                            "name": item,
                            "price": f"${details['price']:.2f}",
                            "in_stock": details['stock'] > 0
                        })

            # Include combos
            for combo_name, combo_details in self.combos.items():
                result["combos"].append({
                    "name": combo_name,
                    "items": combo_details["items"],
                    "price": f"${combo_details['price']:.2f}",
                    "savings": f"${combo_details['savings']:.2f}"
                })

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "confidence": 0.0, "source": "Theater Vendor System"})

    def check_availability(self, items: str) -> str:
        """
        Check inventory availability for specific items.
        Uses fuzzy matching to find closest items (e.g., "caramel popcorn" → "Popcorn Large")

        Args:
            items: Comma-separated list of items (e.g., "Popcorn Large, Soda Medium")
        """
        try:
            item_list = [item.strip() for item in items.split(",")]
            results = []

            for item_query in item_list:
                # Try fuzzy matching
                match = self._fuzzy_match_item(item_query)

                if match:
                    category, item_name, details = match
                    full_name = f"{category} {item_name}"
                    result = {
                        "requested": item_query,
                        "item": full_name,
                        "available": details['stock'] > 0,
                        "stock": details['stock'],
                        "price": f"${details['price']:.2f}"
                    }

                    # Add note if fuzzy matched
                    if item_query.lower() != full_name.lower():
                        result["note"] = f"Matched '{item_query}' to '{full_name}'"

                    results.append(result)
                else:
                    results.append({
                        "requested": item_query,
                        "available": False,
                        "error": "Item not found"
                    })

            return json.dumps({
                "availability": results,
                "confidence": 1.0,
                "source": "Theater Vendor Inventory System"
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "confidence": 0.0, "source": "Theater Vendor Inventory System"})

    def place_order(self, items: str, pickup_time: Optional[str] = None) -> str:
        """
        Place an order for snacks.
        Uses fuzzy matching to find closest items (e.g., "caramel popcorn" → "Popcorn Large")

        Args:
            items: Comma-separated list (e.g., "Popcorn Large x2, Soda Medium")
            pickup_time: Preferred pickup time (optional)
        """
        try:
            item_list = [item.strip() for item in items.split(",")]
            order_items = []
            total_price = 0.0

            for item_query in item_list:
                # Parse quantity (e.g., "Popcorn Large x2")
                quantity = 1
                if " x" in item_query:
                    parts = item_query.rsplit(" x", 1)
                    item_query = parts[0].strip()
                    quantity = int(parts[1].strip())

                # Use fuzzy matching to find item
                match = self._fuzzy_match_item(item_query)

                if not match:
                    return json.dumps({
                        "error": f"Item '{item_query}' not found",
                        "confidence": 0.0,
                        "source": "Theater Vendor Order System"
                    })

                category, item_name, details = match
                full_name = f"{category} {item_name}"

                # Check stock
                if details['stock'] < quantity:
                    # Find substitution suggestions
                    substitutions = self._find_substitutions(category, item_name, quantity)

                    error_response = {
                        "error": "Insufficient inventory",
                        "item": full_name,
                        "requested": quantity,
                        "available": details['stock'],
                        "confidence": 1.0,
                        "source": "Theater Vendor Order System"
                    }

                    # Add substitutions if available
                    if substitutions:
                        error_response["substitutions"] = substitutions
                        error_response["message"] = f"We don't have enough {full_name} in stock. Consider these alternatives:"

                    return json.dumps(error_response, indent=2)

                # Reserve inventory
                details['stock'] -= quantity
                item_price = details['price'] * quantity
                total_price += item_price

                order_item = {
                    "item": full_name,
                    "quantity": quantity,
                    "unit_price": f"${details['price']:.2f}",
                    "subtotal": f"${item_price:.2f}"
                }

                # Add note if fuzzy matched
                if item_query.lower() != full_name.lower():
                    order_item["note"] = f"Matched '{item_query}' to '{full_name}'"

                order_items.append(order_item)

            # Calculate total
            tax = total_price * 0.08
            final_total = total_price + tax

            # Generate pickup time
            if not pickup_time:
                pickup_time = (datetime.now() + timedelta(minutes=5)).strftime("%H:%M")

            # Create order
            order_id = f"ORD{random.randint(100000, 999999)}"
            pickup_code = f"PICK{random.randint(1000, 9999)}"

            order = {
                "order_id": order_id,
                "items": order_items,
                "subtotal": f"${total_price:.2f}",
                "tax": f"${tax:.2f}",
                "total": f"${final_total:.2f}",
                "pickup_time": pickup_time,
                "pickup_code": pickup_code,
                "status": "confirmed"
            }

            self.orders_db[order_id] = order

            return json.dumps({
                "status": "success",
                "message": "Order placed successfully",
                "order": order,
                "confidence": 1.0,
                "source": "Theater Vendor Order System"
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e), "confidence": 0.0, "source": "Theater Vendor Order System"})

    def create_tools(self):
        """Create LangChain tools from instance methods"""
        @tool
        def get_snack_pricing(
            item_name: Annotated[Optional[str], "Specific item name (optional)"] = None
        ) -> str:
            """Get pricing information for snacks and combos."""
            return self.get_snack_pricing(item_name)

        @tool
        def check_availability(
            items: Annotated[str, "Comma-separated list of items (e.g., 'Popcorn Large, Soda Medium')"]
        ) -> str:
            """Check inventory availability for specific items."""
            return self.check_availability(items)

        @tool
        def place_order(
            items: Annotated[str, "Comma-separated items with quantities (e.g., 'Popcorn Large x2, Soda Medium')"],
            pickup_time: Annotated[Optional[str], "Pickup time in HH:MM format (optional)"] = None
        ) -> str:
            """Place an order for snacks with inventory reservation."""
            return self.place_order(items, pickup_time)

        return [get_snack_pricing, check_availability, place_order]

    def create_agent(self):
        """Create the Vendor agent with exactly 3 tools"""
        tools = self.create_tools()
        assert len(tools) <= 3, "Vendor cannot have more than 3 tools"

        system_prompt = """You are the Vendor/Store Manager at a movie theater.

Your responsibilities:
- Provide snack pricing and menu information
- Check inventory availability for items
- Process snack orders and provide pickup details

You have access to exactly 3 tools:
1. get_snack_pricing: Get prices for snacks and combo deals
2. check_availability: Check stock status for items
3. place_order: Process orders with inventory reservation

Important guidelines:
- ALWAYS cite confidence scores and sources for EVERY piece of information
- Never hallucinate prices, stock levels, or order details not returned by your tools
- Determine if the query is within your domain (snacks, food, beverages, orders)
  * IN-DOMAIN: snack prices, inventory, combos, food orders, pickup times, ordering requests
  * OUT-OF-DOMAIN: movie recommendations, actor info, tickets, showtimes, reservations
- If asked about OUT-OF-DOMAIN topics, prefix your response with the keyword OUT-OF-DOMAIN, and
 politely inform the user this is outside your expertise
- Suggest combo deals when they save money
- Ground all information in actual returned data

CRITICAL: When users ask to ORDER items:
- This is IN-DOMAIN - use the place_order tool to process the order
- Do NOT respond with OUT-OF-DOMAIN for ordering requests
- Default to Large size if no size is specified
- Default quantity to 1 if not specified

CRITICAL Validation requirements:
- MUST include confidence scores from tool responses in your final answer
- MUST cite data sources (Theater Vendor System) for each piece of information
- Format: Always include "Confidence: X%" and "Source: [source name]" for each response

Example response format:
"A Large Popcorn costs $9.00 and a Medium Soda costs $5.50. Both are in stock.
**Confidence: 100%** | **Source: Theater Vendor System**"
"""

        agent = create_agent(llm, tools, system_prompt=system_prompt)
        return agent

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, Exception))
    )
    def _invoke_with_retry(self, messages: List) -> Dict[str, Any]:
        """Invoke agent with retry logic and timeout

        Args:
            messages: List of messages for the agent

        Returns:
            Agent result dictionary
        """
        return self.agent.invoke({"messages": messages})

    def invoke(self, query: str, state: Optional[Dict[str, Any]] = None) -> str:
        """Invoke the Vendor agent with a query

        Args:
            query: User's question or request
            state: Optional session state (for context preservation)

        Returns:
            str: Agent's response content
        """
        # Add user message to history
        self.message_history.append(HumanMessage(content=query))

        try:
            # Invoke agent with full conversation history and retry logic
            result = self._invoke_with_retry(self.message_history)

            # Update history with all messages from result
            self.message_history = result["messages"]

            # Return the last message content
            agent_response = result["messages"][-1].content if result["messages"] else "No response"
            return agent_response

        except Exception as e:
            # If all retries fail, return error message
            error_msg = f"I apologize, but I encountered an error processing your vendor request. Error: {str(e)[:200]}. Please try again."
            # Add error as AI message to maintain conversation flow
            self.message_history.append(AIMessage(content=error_msg))
            return error_msg


if __name__ == "__main__":
    print("\nVENDOR AGENT - Demo\n")

    vendor = VendorAgent()

    test_queries = [
        ("Get Snack Pricing", "How much is a large popcorn and soda?"),
        ("Check Availability", "Do you have nachos available?"),
        ("Place Order", "I want to order 2 large popcorns and a medium soda"),
        ("Fuzzy Match Order", "Can you add a large caramel popcorn?"),
        ("Simple Order", "Order popcorn"),
        ("OUT-OF-DOMAIN: Movies", "What movies are playing tonight?"),
    ]

    for title, query in test_queries:
        print("=" * 80)
        print(f"{title}")
        print("=" * 80)
        print(f"Query: {query}")
        response = vendor.invoke(query)
        print(f"Response: {response}\n")