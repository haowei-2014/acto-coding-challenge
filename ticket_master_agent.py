"""
Ticket Master Agent

Handles showtimes, pricing, and ticket purchases.
Max 3 tools enforced.
"""

import json
import os
from typing import Annotated, List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Structured output schema
class TicketMasterOutput(TypedDict):
    """Structured output from Ticket Master Agent"""
    competent: bool  # If user's query is out of domain, return False. Otherwise return True
    content: str  # Natural language response with confidence and source


class TicketMasterAgent:
    """Ticket Master Agent - handles showtimes, pricing, and reservations"""

    def __init__(self):
        """Initialize the Ticket Master with showtime database"""
        self.showtimes_db = {}
        self.reservations_db = {}
        self.movies_list = []
        self._initialize_showtimes()

    def _initialize_showtimes(self):
        """Initialize showtime database based on movie data"""
        if self.showtimes_db:
            return  # Already initialized

        # Load movie data
        try:
            movies_df = pd.read_csv('tmdb_5000_movies_small.csv', delimiter=';')

            # Get top 20 popular movies
            movies_df = movies_df.sort_values('popularity', ascending=False).head(20)
            self.movies_list = movies_df['title'].tolist()

            # Generate showtimes for next 7 days
            base_date = datetime.now()
            formats = ['Standard', 'IMAX', '3D', 'IMAX 3D']
            time_slots = ['10:00', '13:00', '16:00', '19:00', '22:00']

            for movie_title in self.movies_list:
                self.showtimes_db[movie_title] = []

                for day_offset in range(7):
                    date = base_date + timedelta(days=day_offset)
                    date_str = date.strftime('%Y-%m-%d')

                    # Random formats available for this movie on this day
                    available_formats = random.sample(formats, k=random.randint(2, 4))

                    for fmt in available_formats:
                        # Random time slots for this format
                        available_times = random.sample(time_slots, k=random.randint(2, 4))

                        for time in available_times:
                            showtime_id = f"{movie_title}_{date_str}_{time}_{fmt}".replace(" ", "_")
                            self.showtimes_db[movie_title].append({
                                'showtime_id': showtime_id,
                                'movie': movie_title,
                                'date': date_str,
                                'time': time,
                                'format': fmt,
                                'available_seats': random.randint(50, 200),
                                'total_seats': 200
                            })

            print(f"Initialized showtimes for {len(self.movies_list)} movies")
        except Exception as e:
            print(f"Error initializing showtimes: {e}")
            # Create minimal test data
            self.movies_list = ['Avatar', 'Inception', 'The Dark Knight']
            for movie in self.movies_list:
                self.showtimes_db[movie] = [{
                    'showtime_id': f"{movie}_2025-10-20_19:00_IMAX".replace(" ", "_"),
                    'movie': movie,
                    'date': '2025-10-20',
                    'time': '19:00',
                    'format': 'IMAX',
                    'available_seats': 150,
                    'total_seats': 200
                }]

    def check_showtimes(
        self,
        movie_title: str,
        date: Optional[str] = None,
        format_preference: Optional[str] = None
    ) -> str:
        """
        Check available showtimes for a specific movie.
        Returns date, time, format, and seat availability.
        """
        try:
            # Find movie (case-insensitive)
            movie_key = None
            for movie in self.showtimes_db.keys():
                if movie.lower() == movie_title.lower() or movie_title.lower() in movie.lower():
                    movie_key = movie
                    break

            if not movie_key:
                return json.dumps({
                    "error": f"Movie '{movie_title}' not found in current listings",
                    "available_movies": self.movies_list[:10],
                    "confidence": 0.0,
                    "source": "Theater Showtime Database"
                })

            showtimes = self.showtimes_db[movie_key]

            # Filter by date if provided
            if date:
                showtimes = [s for s in showtimes if s['date'] == date]

            # Filter by format if provided
            if format_preference:
                showtimes = [s for s in showtimes if format_preference.lower() in s['format'].lower()]

            if not showtimes:
                return json.dumps({
                    "error": f"No showtimes found for '{movie_key}' with given criteria",
                    "date_filter": date,
                    "format_filter": format_preference,
                    "confidence": 0.0,
                    "source": "Theater Showtime Database"
                })

            # Group by date
            showtimes_by_date = {}
            for showtime in showtimes:
                date_key = showtime['date']
                if date_key not in showtimes_by_date:
                    showtimes_by_date[date_key] = []
                showtimes_by_date[date_key].append({
                    "showtime_id": showtime['showtime_id'],
                    "time": showtime['time'],
                    "format": showtime['format'],
                    "available_seats": showtime['available_seats'],
                    "status": "Available" if showtime['available_seats'] > 10 else "Limited Seats"
                })

            return json.dumps({
                "movie": movie_key,
                "showtimes": showtimes_by_date,
                "total_showtimes": len(showtimes),
                "confidence": 1.0,
                "source": "Theater Showtime Database"
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "confidence": 0.0,
                "source": "Theater Showtime Database"
            })

    def get_pricing(
        self,
        format_type: str,
        party_size: int = 1,
        discount_code: Optional[str] = None
    ) -> str:
        """
        Get pricing information for tickets including discounts and fees.
        Returns detailed price breakdown.
        """
        try:
            # Base pricing
            base_prices = {
                'standard': 12.00,
                'imax': 18.00,
                '3d': 15.00,
                'imax 3d': 22.00
            }

            # Find matching format
            format_key = format_type.lower()
            base_price = base_prices.get(format_key, base_prices['standard'])

            # Calculate subtotal
            subtotal = base_price * party_size

            # Apply discount if provided
            discount_amount = 0.0
            discount_description = "None"
            if discount_code:
                discounts = {
                    'STUDENT10': 0.10,
                    'SENIOR20': 0.20,
                    'FAMILY15': 0.15,
                    'EARLYBIRD': 0.25
                }
                if discount_code.upper() in discounts:
                    discount_rate = discounts[discount_code.upper()]
                    discount_amount = subtotal * discount_rate
                    discount_description = f"{discount_code.upper()} ({int(discount_rate * 100)}% off)"

            # Calculate fees and taxes
            service_fee = party_size * 1.50
            tax_rate = 0.08
            subtotal_after_discount = subtotal - discount_amount
            tax = subtotal_after_discount * tax_rate
            total = subtotal_after_discount + service_fee + tax

            pricing = {
                "format": format_type,
                "party_size": party_size,
                "base_price_per_ticket": f"${base_price:.2f}",
                "subtotal": f"${subtotal:.2f}",
                "discount": {
                    "code": discount_code if discount_code else "None",
                    "description": discount_description,
                    "amount": f"-${discount_amount:.2f}"
                },
                "service_fee": f"${service_fee:.2f}",
                "tax": f"${tax:.2f}",
                "total": f"${total:.2f}",
                "breakdown": [
                    f"Base: {party_size} x ${base_price:.2f} = ${subtotal:.2f}",
                    f"Discount: -{discount_amount:.2f}",
                    f"Service Fee: ${service_fee:.2f}",
                    f"Tax (8%): ${tax:.2f}",
                    f"Total: ${total:.2f}"
                ],
                "confidence": 1.0,
                "source": "Theater Pricing System"
            }

            return json.dumps(pricing, indent=2)

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "confidence": 0.0,
                "source": "Theater Pricing System"
            })

    def purchase_tickets(
        self,
        showtime_id: str,
        party_size: int,
        payment_token: str = "mock_payment_token"
    ) -> str:
        """
        Reserve or purchase tickets for a specific showtime.
        Returns reservation confirmation with details.
        Implements two-phase: reserve → confirm with idempotency.
        """
        try:
            # Find the showtime
            showtime = None
            for movie, times in self.showtimes_db.items():
                for st in times:
                    if st['showtime_id'] == showtime_id:
                        showtime = st
                        break
                if showtime:
                    break

            if not showtime:
                return json.dumps({
                    "error": f"Showtime ID '{showtime_id}' not found",
                    "confidence": 0.0,
                    "source": "Theater Reservation System"
                })

            # Check idempotency - if already reserved
            if showtime_id in self.reservations_db:
                existing = self.reservations_db[showtime_id]
                return json.dumps({
                    "status": "already_reserved",
                    "message": "This showtime was already reserved",
                    "reservation": existing,
                    "confidence": 1.0,
                    "source": "Theater Reservation System"
                }, indent=2)

            # Phase 1: Check availability
            if showtime['available_seats'] < party_size:
                return json.dumps({
                    "error": "Not enough seats available",
                    "requested": party_size,
                    "available": showtime['available_seats'],
                    "confidence": 1.0,
                    "source": "Theater Reservation System"
                })

            # Phase 2: Reserve seats
            showtime['available_seats'] -= party_size

            # Create reservation
            reservation_id = f"RES{random.randint(100000, 999999)}"
            reservation = {
                "reservation_id": reservation_id,
                "showtime_id": showtime_id,
                "movie": showtime['movie'],
                "date": showtime['date'],
                "time": showtime['time'],
                "format": showtime['format'],
                "party_size": party_size,
                "seats_reserved": party_size,
                "payment_token": payment_token,
                "status": "confirmed",
                "confirmation_code": f"CONF{random.randint(1000, 9999)}"
            }

            # Store reservation (idempotency key: showtime_id)
            self.reservations_db[showtime_id] = reservation

            return json.dumps({
                "status": "success",
                "message": f"Successfully reserved {party_size} ticket(s)",
                "reservation": reservation,
                "remaining_seats": showtime['available_seats'],
                "confidence": 1.0,
                "source": "Theater Reservation System"
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "confidence": 0.0,
                "source": "Theater Reservation System"
            })

    def create_tools(self):
        """Create LangChain tools from instance methods"""
        @tool
        def check_showtimes(
            movie_title: Annotated[str, "Movie title to check showtimes for"],
            date: Annotated[Optional[str], "Date in YYYY-MM-DD format (optional)"] = None,
            format_preference: Annotated[Optional[str], "Format preference: Standard, IMAX, 3D, IMAX 3D (optional)"] = None
        ) -> str:
            """
            Check available showtimes for a specific movie.
            Returns date, time, format, and seat availability.
            """
            return self.check_showtimes(movie_title, date, format_preference)

        @tool
        def get_pricing(
            format_type: Annotated[str, "Format type: Standard, IMAX, 3D, or IMAX 3D"],
            party_size: Annotated[int, "Number of tickets"] = 1,
            discount_code: Annotated[Optional[str], "Discount code (optional)"] = None
        ) -> str:
            """
            Get pricing information for tickets including discounts and fees.
            Returns detailed price breakdown.
            """
            return self.get_pricing(format_type, party_size, discount_code)

        @tool
        def purchase_tickets(
            showtime_id: Annotated[str, "Showtime ID from check_showtimes"],
            party_size: Annotated[int, "Number of tickets to purchase"],
            payment_token: Annotated[str, "Payment token (mock)"] = "mock_payment_token"
        ) -> str:
            """
            Reserve or purchase tickets for a specific showtime.
            Returns reservation confirmation with details.
            Implements two-phase: reserve → confirm with idempotency.
            """
            return self.purchase_tickets(showtime_id, party_size, payment_token)

        return [check_showtimes, get_pricing, purchase_tickets]

    def create_agent(self):
        """Create the Ticket Master agent with exactly 3 tools"""
        tools = self.create_tools()

        # Verify tool limit
        assert len(tools) <= 3, "Ticket Master cannot have more than 3 tools"

        # System prompt for Ticket Master
        system_prompt = """You are the Ticket Master at a movie theater.

Your responsibilities:
- Check showtime availability for movies
- Provide ticket pricing information with discounts
- Process ticket reservations and purchases

You have access to exactly 3 tools:
1. check_showtimes: Find available showtimes by movie, date, and format
2. get_pricing: Calculate ticket prices with discounts and fees
3. purchase_tickets: Reserve and confirm ticket purchases

Important guidelines:
- ALWAYS cite confidence scores and sources for EVERY piece of information
- Never hallucinate showtimes, prices, or reservation details not returned by your tools
- Determine if the query is within your domain (tickets, showtimes, pricing, purchases)
  * IN-DOMAIN: showtimes, ticket prices, reservations, seat availability, formats
  * OUT-OF-DOMAIN: movie recommendations, actor info, plot details, snacks, vendor items
- If asked about OUT-OF-DOMAIN topics, politely inform the user this is outside your expertise
- Always re-check seat availability before confirming purchases
- Provide clear price breakdowns showing all fees and taxes
- Ground all information in actual returned data

CRITICAL Validation requirements:
- MUST include confidence scores from tool responses in your final answer
- MUST cite data sources (Theater Database/System) for each piece of information
- Ground all recommendations in actual returned data
- No made-up showtimes, prices, or reservations
- Format: Always include "Confidence: X%" and "Source: [source name]" for each response

Example response format:
"I found showtimes for Avatar on 2025-10-20 at 19:00 in IMAX format with 150 seats available.
**Confidence: 100%** | **Source: Theater Showtime Database**"
"""

        # Create agent with system prompt
        agent = create_react_agent(
            llm,
            tools,
            prompt=system_prompt
        )

        return agent

    def invoke(self, query: str, state: Optional[Dict[str, Any]] = None) -> TicketMasterOutput:
        """
        Invoke the Ticket Master agent with a query

        Args:
            query: User's question or request
            state: Optional session state (for context preservation)

        Returns:
            TicketMasterOutput with competent flag and content
        """
        agent = self.create_agent()

        # Prepare messages
        messages = [HumanMessage(content=query)]

        # Invoke agent
        result = agent.invoke({"messages": messages})

        # Extract response content
        agent_response = result["messages"][-1].content if result["messages"] else "No response"

        # Determine competency based on response
        # Check if the response indicates out-of-domain
        out_of_domain_keywords = [
            "outside my expertise", "not my area", "movie specialist", "vendor",
            "can't help with recommendations", "can't help with actor", "plot",
            "movie details", "actor information"
        ]

        competent = not any(keyword.lower() in agent_response.lower() for keyword in out_of_domain_keywords)

        # Return structured output
        return TicketMasterOutput(
            competent=competent,
            content=agent_response
        )


# Global instance (singleton pattern)
_ticket_master_instance = None


def get_ticket_master() -> TicketMasterAgent:
    """Get or create the global Ticket Master instance"""
    global _ticket_master_instance
    if _ticket_master_instance is None:
        _ticket_master_instance = TicketMasterAgent()
    return _ticket_master_instance


def invoke_ticket_master(query: str, state: Optional[Dict[str, Any]] = None) -> TicketMasterOutput:
    """
    Convenience function to invoke the Ticket Master agent

    Args:
        query: User's question or request
        state: Optional session state (for context preservation)

    Returns:
        TicketMasterOutput with competent flag and content
    """
    ticket_master = get_ticket_master()
    return ticket_master.invoke(query, state)


if __name__ == "__main__":
    # Test the agent
    print("Testing Ticket Master Agent with Structured Output...\n")

    # Test 1: In-domain query - Check Showtimes
    print("=" * 80)
    print("Test 1: Check Showtimes (IN-DOMAIN)")
    print("=" * 80)
    query1 = "What showtimes are available for Avatar?"
    response1 = invoke_ticket_master(query1)
    print(f"Query: {query1}")
    print(f"Competent: {response1['competent']}")
    print(f"Response: {response1['content']}\n")

    # Test 2: In-domain query - Get Pricing
    print("=" * 80)
    print("Test 2: Get Pricing (IN-DOMAIN)")
    print("=" * 80)
    query2 = "How much for 2 IMAX tickets?"
    response2 = invoke_ticket_master(query2)
    print(f"Query: {query2}")
    print(f"Competent: {response2['competent']}")
    print(f"Response: {response2['content']}\n")

    # Test 3: In-domain query - Purchase Tickets
    print("=" * 80)
    print("Test 3: Purchase Tickets (IN-DOMAIN)")
    print("=" * 80)
    query3 = "Can I reserve 2 tickets for Avatar at 7pm in IMAX?"
    response3 = invoke_ticket_master(query3)
    print(f"Query: {query3}")
    print(f"Competent: {response3['competent']}")
    print(f"Response: {response3['content']}\n")

    # Test 4: Out-of-domain query - Movie Recommendations
    print("=" * 80)
    print("Test 4: Movie Recommendations (OUT-OF-DOMAIN)")
    print("=" * 80)
    query4 = "What's a good sci-fi movie to watch?"
    response4 = invoke_ticket_master(query4)
    print(f"Query: {query4}")
    print(f"Competent: {response4['competent']}")
    print(f"Response: {response4['content']}\n")

    # Test 5: Out-of-domain query - Snacks
    print("=" * 80)
    print("Test 5: Snack Ordering (OUT-OF-DOMAIN)")
    print("=" * 80)
    query5 = "Can I get popcorn with my tickets?"
    response5 = invoke_ticket_master(query5)
    print(f"Query: {query5}")
    print(f"Competent: {response5['competent']}")
    print(f"Response: {response5['content']}\n")
