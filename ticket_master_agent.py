"""
Ticket Master Agent

Handles showtimes, pricing, and ticket purchases.
Max 3 tools enforced.
"""

import json
import os
from typing import Annotated, List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random
import calendar

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)


class TicketMasterAgent:
    """Ticket Master Agent - handles showtimes, pricing, and reservations"""

    def __init__(self):
        """Initialize the Ticket Master with showtime database"""
        self.showtimes_db = {}
        self.reservations_db = {}
        self.movies_list = []
        self._initialize_showtimes()
        self.agent = self.create_agent()  # Create agent once, reuse for conversation history
        self.message_history = []  # Track conversation history

    def _initialize_showtimes(self):
        """Initialize showtime database based on movie data"""
        if self.showtimes_db:
            return  # Already initialized

        # Load movie data
        try:
            # Load first 100 rows only
            movies_df = pd.read_csv('data/tmdb_5000_movies.csv', nrows=100)

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

    def _parse_date(self, date_input: Optional[str]) -> Optional[str]:
        """
        Parse date input which can be:
        - YYYY-MM-DD format
        - Weekday name (e.g., "Monday", "Friday")
        - Relative terms (e.g., "today", "tomorrow")

        Returns date in YYYY-MM-DD format or None
        """
        if not date_input:
            return None

        date_lower = date_input.lower().strip()

        # Check if already in YYYY-MM-DD format
        if '-' in date_input and len(date_input) == 10:
            return date_input

        today = datetime.now()

        # Handle relative terms
        if date_lower in ['today']:
            return today.strftime('%Y-%m-%d')
        elif date_lower in ['tomorrow']:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')

        # Handle weekday names
        weekdays = {
            'monday': 0, 'mon': 0,
            'tuesday': 1, 'tue': 1, 'tues': 1,
            'wednesday': 2, 'wed': 2,
            'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
            'friday': 4, 'fri': 4,
            'saturday': 5, 'sat': 5,
            'sunday': 6, 'sun': 6
        }

        if date_lower in weekdays:
            target_weekday = weekdays[date_lower]
            current_weekday = today.weekday()

            # Calculate days until target weekday
            days_ahead = target_weekday - current_weekday
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7  # Get next week's occurrence

            target_date = today + timedelta(days=days_ahead)
            return target_date.strftime('%Y-%m-%d')

        # If we can't parse it, return as-is and let downstream handle it
        return date_input

    def check_showtimes(
        self,
        movie_title: Optional[str] = None,
        date: Optional[str] = None,
        format_preference: Optional[str] = None
    ) -> str:
        """
        Check available showtimes. Can filter by movie, date, and/or format.
        If movie_title is not provided, returns showtimes for all movies matching other criteria.
        Date can be: YYYY-MM-DD format, weekday name (Monday-Sunday), or relative terms (today/tomorrow).
        Returns date, time, format, and seat availability.
        """
        try:
            # Parse date input (convert weekdays to YYYY-MM-DD)
            parsed_date = self._parse_date(date)

            # Collect all matching showtimes
            all_showtimes = []

            if movie_title:
                # Find specific movie (case-insensitive)
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

                # Get showtimes for specific movie
                for showtime in self.showtimes_db[movie_key]:
                    all_showtimes.append({**showtime, 'movie': movie_key})
            else:
                # Get showtimes for all movies
                for movie, showtimes in self.showtimes_db.items():
                    for showtime in showtimes:
                        all_showtimes.append({**showtime, 'movie': movie})

            # Filter by date if provided (use parsed_date)
            if parsed_date:
                all_showtimes = [s for s in all_showtimes if s['date'] == parsed_date]

            # Filter by format if provided
            if format_preference:
                all_showtimes = [s for s in all_showtimes if format_preference.lower() in s['format'].lower()]

            if not all_showtimes:
                return json.dumps({
                    "error": "No showtimes found matching the criteria",
                    "movie_filter": movie_title,
                    "date_filter": f"{date} (parsed as: {parsed_date})" if parsed_date != date else date,
                    "format_filter": format_preference,
                    "confidence": 0.0,
                    "source": "Theater Showtime Database"
                })

            # Group by movie and date
            showtimes_grouped = {}
            for showtime in all_showtimes:
                movie_name = showtime['movie']
                if movie_name not in showtimes_grouped:
                    showtimes_grouped[movie_name] = {}

                date_key = showtime['date']
                if date_key not in showtimes_grouped[movie_name]:
                    showtimes_grouped[movie_name][date_key] = []

                showtimes_grouped[movie_name][date_key].append({
                    "showtime_id": showtime['showtime_id'],
                    "time": showtime['time'],
                    "format": showtime['format'],
                    "available_seats": showtime['available_seats'],
                    "status": "Available" if showtime['available_seats'] > 10 else "Limited Seats"
                })

            # Format response based on whether movie was specified
            if movie_title:
                # Single movie response
                movie_key = list(showtimes_grouped.keys())[0]
                return json.dumps({
                    "movie": movie_key,
                    "showtimes": showtimes_grouped[movie_key],
                    "total_showtimes": sum(len(times) for times in showtimes_grouped[movie_key].values()),
                    "confidence": 1.0,
                    "source": "Theater Showtime Database"
                }, indent=2)
            else:
                # Multiple movies response
                movies_list = []
                for movie_name, dates in showtimes_grouped.items():
                    total_showtimes = sum(len(times) for times in dates.values())
                    movies_list.append({
                        "movie": movie_name,
                        "showtimes": dates,
                        "total_showtimes": total_showtimes
                    })

                response_data = {
                    "movies": movies_list,
                    "total_movies": len(movies_list),
                    "filters_applied": {
                        "date": parsed_date,
                        "format": format_preference
                    },
                    "confidence": 1.0,
                    "source": "Theater Showtime Database"
                }

                # Add date interpretation note if date was parsed from weekday
                if date and parsed_date and date.lower() != parsed_date:
                    response_data["date_interpretation"] = f"'{date}' interpreted as {parsed_date}"

                return json.dumps(response_data, indent=2)

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
            movie_title: Annotated[Optional[str], "Movie title to check (optional - if not provided, shows all movies)"] = None,
            date: Annotated[Optional[str], "Date: can be YYYY-MM-DD, weekday name (Monday-Sunday), today, or tomorrow (optional)"] = None,
            format_preference: Annotated[Optional[str], "Format preference: Standard, IMAX, 3D, IMAX 3D (optional)"] = None
        ) -> str:
            """
            Check available showtimes. Can filter by movie title, date, and/or format.
            If no movie_title is provided, returns all available movies matching the other criteria.
            Date accepts: YYYY-MM-DD format, weekday names (e.g., Friday), or relative terms (today/tomorrow).
            Returns date, time, format, and seat availability for each match.
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
- ALWAYS use your tools to answer queries - don't ask for clarification when tools can handle optional parameters
- ALWAYS cite confidence scores and sources for EVERY piece of information
- Never hallucinate showtimes, prices, or reservation details not returned by your tools
- Determine if the query is within your domain (tickets, showtimes, pricing, purchases)
  * IN-DOMAIN: showtimes, ticket prices, reservations, seat availability, formats
  * OUT-OF-DOMAIN: movie recommendations, actor info, plot details, snacks, vendor items
- If asked about OUT-OF-DOMAIN topics, prefix your response with the keyword OUT-OF-DOMAIN, and
  politely inform the user this is outside your expertise
- Always re-check seat availability before confirming purchases
- Provide clear price breakdowns showing all fees and taxes
- Ground all information in actual returned data

Workflow for ticket purchase requests:
1. If user asks for tickets, first check_showtimes to find available showtimes
2. If user wants to proceed with purchase, call purchase_tickets with the showtime_id
3. DO NOT call get_pricing separately - purchase_tickets already includes pricing
4. After completing a purchase, provide the confirmation and STOP - do not call more tools

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
        return self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 25}
        )

    def invoke(self, query: str, state: Optional[Dict[str, Any]] = None) -> str:
        """
        Invoke the Ticket Master agent with a query

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
            error_msg = f"I apologize, but I encountered an issue processing your ticket request. Error: {str(e)[:200]}. Please provide more specific details like the movie title, date, and time."
            # Add error as AI message to maintain conversation flow
            self.message_history.append(AIMessage(content=error_msg))
            return error_msg



if __name__ == "__main__":
    print("\nTICKET MASTER AGENT - Demo\n")

    ticket_master = TicketMasterAgent()

    test_queries = [
        ("Check Showtimes", "What showtimes are available for Avatar?"),
        ("Get Pricing", "How much for 2 IMAX tickets?"),
        ("Purchase Tickets", "Can I reserve 2 tickets for Avatar at 7pm in IMAX?"),
        ("Showtimes Without Movie", "Can I get two IMAX tickets for Friday 19:30?"),
        ("OUT-OF-DOMAIN: Movies", "What's a good sci-fi movie to watch?"),
        ("OUT-OF-DOMAIN: Snacks", "Can I get popcorn with my tickets?"),
    ]

    for title, query in test_queries:
        print("=" * 80)
        print(f"{title}")
        print("=" * 80)
        print(f"Query: {query}")
        response = ticket_master.invoke(query)
        print(f"Response: {response}\n")