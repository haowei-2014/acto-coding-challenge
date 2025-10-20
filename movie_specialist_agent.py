"""
Movie Specialist Agent

Handles movie recommendations, details, and actor information.
Max 3 tools enforced.
"""

import json
import os
from typing import Annotated, List, Dict, Any, Optional
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()


class MovieSpecialistAgent:
    """Movie Specialist Agent - handles movie recommendations, details, and actor information"""

    def __init__(self):
        """Initialize the Movie Specialist with movie database"""
        self.movies_df = None
        self.credits_df = None
        self._load_movie_data()
        self.agent = self.create_agent()  # Create agent once, reuse for conversation history
        self.message_history = []  # Track conversation history

    def _load_movie_data(self):
        """Load and prepare movie data"""
        if self.movies_df is not None:
            return  # Already loaded

        try:
            # Load datasets (first 100 rows only)
            self.movies_df = pd.read_csv('data/tmdb_5000_movies.csv', nrows=100)
            self.credits_df = pd.read_csv('data/tmdb_5000_credits.csv', nrows=100)

            # Parse JSON columns
            self.movies_df['genres_parsed'] = self.movies_df['genres'].apply(
                lambda x: [g['name'] for g in json.loads(x)] if pd.notna(x) else []
            )
            self.movies_df['keywords_parsed'] = self.movies_df['keywords'].apply(
                lambda x: [k['name'] for k in json.loads(x)] if pd.notna(x) else []
            )

            # Parse cast from credits
            self.credits_df['cast_parsed'] = self.credits_df['cast'].apply(
                lambda x: json.loads(x) if pd.notna(x) else []
            )

            # Merge credits with movies
            self.movies_df = self.movies_df.merge(
                self.credits_df[['movie_id', 'cast_parsed']],
                left_on='id', right_on='movie_id', how='left'
            )

            # Create searchable text for embeddings
            self.movies_df['search_text'] = self.movies_df.apply(
                lambda row: f"{row['title']}. {' '.join(row['genres_parsed'])}. {row['overview']}. "
                           f"Keywords: {' '.join(row['keywords_parsed'][:10])}",
                axis=1
            )

            print(f"Loaded {len(self.movies_df)} movies")
        except Exception as e:
            print(f"Error loading movie data: {e}")

    def recommend_movies(self, query: str, limit: int = 5) -> str:
        """Find movie recommendations based on semantic similarity"""
        try:
            # Get embedding for user query
            query_embedding = embeddings.embed_query(query)

            # Get or create movie embeddings (cache for efficiency)
            # For demo, we'll compute a subset
            sample_size = min(50, len(self.movies_df))
            sample_df = self.movies_df.head(sample_size).copy()

            # Compute embeddings for sample
            movie_texts = sample_df['search_text'].tolist()
            movie_embeddings_list = embeddings.embed_documents(movie_texts)

            # Calculate cosine similarity
            query_vec = np.array(query_embedding)
            movie_vecs = np.array(movie_embeddings_list)

            similarities = np.dot(movie_vecs, query_vec) / (
                np.linalg.norm(movie_vecs, axis=1) * np.linalg.norm(query_vec)
            )

            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:limit]

            recommendations = []
            for idx in top_indices:
                movie = sample_df.iloc[idx]
                confidence = float(similarities[idx])

                rec = {
                    "title": movie['title'],
                    "year": movie['release_date'][:4] if pd.notna(movie['release_date']) else "N/A",
                    "genres": movie['genres_parsed'],
                    "rating": float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0,
                    "overview": movie['overview'][:150] + "..." if pd.notna(movie['overview']) and len(movie['overview']) > 150 else movie['overview'],
                    "confidence": round(confidence, 3),
                    "reason": f"Matches {', '.join(movie['genres_parsed'][:3])} genres with {int(confidence * 100)}% confidence",
                    "source": "TMDB Dataset - embedding similarity"
                }
                recommendations.append(rec)

            return json.dumps({
                "query": query,
                "recommendations": recommendations,
                "total_found": len(recommendations)
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "query": query,
                "recommendations": []
            })
    def get_movie_details(self, title: str) -> str:
        """Get detailed information about a specific movie"""
        try:
            # Case-insensitive search
            matches = self.movies_df[self.movies_df['title'].str.lower() == title.lower()]

            if matches.empty:
                # Try partial match
                matches = self.movies_df[self.movies_df['title'].str.contains(title, case=False, na=False)]

            if matches.empty:
                return json.dumps({
                    "error": f"Movie '{title}' not found in database",
                    "confidence": 0.0,
                    "source": "TMDB Dataset"
                })

            # Get first match
            movie = matches.iloc[0]

            details = {
                "title": movie['title'],
                "original_title": movie['original_title'] if pd.notna(movie['original_title']) else movie['title'],
                "overview": movie['overview'] if pd.notna(movie['overview']) else "No overview available",
                "genres": movie['genres_parsed'],
                "release_date": movie['release_date'] if pd.notna(movie['release_date']) else "Unknown",
                "runtime": f"{int(movie['runtime'])} minutes" if pd.notna(movie['runtime']) else "Unknown",
                "rating": float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0,
                "vote_count": int(movie['vote_count']) if pd.notna(movie['vote_count']) else 0,
                "budget": f"${int(movie['budget']):,}" if pd.notna(movie['budget']) and movie['budget'] > 0 else "Unknown",
                "revenue": f"${int(movie['revenue']):,}" if pd.notna(movie['revenue']) and movie['revenue'] > 0 else "Unknown",
                "language": movie['original_language'] if pd.notna(movie['original_language']) else "Unknown",
                "status": movie['status'] if pd.notna(movie['status']) else "Unknown",
                "tagline": movie['tagline'] if pd.notna(movie['tagline']) else "",
                "keywords": movie['keywords_parsed'][:10],
                "confidence": 1.0 if matches.iloc[0]['title'].lower() == title.lower() else 0.8,
                "source": "TMDB Dataset - exact match" if matches.iloc[0]['title'].lower() == title.lower() else "TMDB Dataset - partial match"
            }

            return json.dumps(details, indent=2)

        except Exception as e:
            return json.dumps({
                "error": str(e),
                "confidence": 0.0,
                "source": "TMDB Dataset"
            })
    def get_actor_info(self, actor_name: str) -> str:
        """Get information about an actor including their filmography"""

        try:
            # Search across all cast lists
            actor_movies = []

            # Check if cast_parsed exists
            if 'cast_parsed' not in self.movies_df.columns:
                return json.dumps({
                    "error": "cast_parsed column not found in dataframe",
                    "available_columns": self.movies_df.columns.tolist(),
                    "confidence": 0.0,
                    "source": "TMDB Dataset"
                })

            for idx, row in self.movies_df.iterrows():
                try:
                    cast_data = row['cast_parsed']
                    # Check if cast_data is a valid list (not NaN, not empty)
                    if isinstance(cast_data, list) and len(cast_data) > 0:
                        for cast_member in cast_data:
                            if isinstance(cast_member, dict) and actor_name.lower() in cast_member.get('name', '').lower():
                                actor_movies.append({
                                    'movie_title': row['title'],
                                    'character': cast_member.get('character', 'Unknown'),
                                    'order': cast_member.get('order', 999),
                                    'release_date': row['release_date'] if pd.notna(row['release_date']) else "Unknown",
                                    'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0,
                                    'overview': row['overview'] if pd.notna(row['overview']) else "",
                                    'actor_full_name': cast_member.get('name', actor_name)
                                })
                except Exception as row_error:
                    print(f"Error processing row {idx}: {row_error}")
                    continue

            if not actor_movies:
                return json.dumps({
                    "error": f"Actor '{actor_name}' not found in database",
                    "confidence": 0.0,
                    "source": "TMDB Dataset"
                })

            print(f"Total actor_movies found: {len(actor_movies)}")

            # Sort by rating and role prominence (order)
            actor_movies.sort(key=lambda x: (x['rating'], -x['order']), reverse=True)

            # Get top films
            top_films = actor_movies[:10]

            print(f"Building actor_info...")

            actor_info = {
                "name": actor_movies[0]['actor_full_name'],
                "total_films_in_database": len(actor_movies),
                "top_films": [
                    {
                        "title": film['movie_title'],
                        "character": film['character'],
                        "year": film['release_date'][:4] if film['release_date'] != "Unknown" else "Unknown",
                        "rating": film['rating'],
                        "billing_order": film['order']
                    }
                    for film in top_films
                ],
                "notable_characters": list(set([f['character'] for f in top_films[:5]])),
                "confidence": 1.0,
                "source": "TMDB Dataset - cast information"
            }

            print(f"Returning actor info...")
            return json.dumps(actor_info, indent=2)

        except Exception as e:
            print("*** exception")
            import traceback
            error_details = traceback.format_exc()
            return json.dumps({
                "error": str(e),
                "error_details": error_details,
                "confidence": 0.0,
                "source": "TMDB Dataset"
            })

    def create_tools(self):
        """Create LangChain tools from instance methods"""
        @tool
        def recommend_movies(
            query: Annotated[str, "User's movie preferences or description"],
            limit: Annotated[int, "Maximum number of recommendations"] = 5
        ) -> str:
            """Find movie recommendations based on semantic similarity to user preferences."""
            return self.recommend_movies(query, limit)

        @tool
        def get_movie_details(
            title: Annotated[str, "Movie title to look up"]
        ) -> str:
            """Get detailed information about a specific movie including plot, genres, ratings, runtime, and release date."""
            return self.get_movie_details(title)

        @tool
        def get_actor_info(
            actor_name: Annotated[str, "Actor's name to look up"]
        ) -> str:
            """Get information about an actor including their filmography highlights and notable roles."""
            return self.get_actor_info(actor_name)

        return [recommend_movies, get_movie_details, get_actor_info]

    def create_agent(self):
        """Create the Movie Specialist agent with exactly 3 tools"""
        tools = self.create_tools()
        assert len(tools) <= 3, "Movie Specialist cannot have more than 3 tools"

        system_prompt = """You are the Movie Specialist at a movie theater.

Your responsibilities:
- Provide movie recommendations based on user preferences (genre, mood, themes)
- Share detailed information about movies (plot, ratings, runtime, etc.)
- Provide actor information and their filmography highlights

You have access to exactly 3 tools:
1. recommend_movies: Find movies matching user preferences using semantic similarity
2. get_movie_details: Get detailed information about a specific movie
3. get_actor_info: Get actor details and their top films

IMPORTANT - Use Session Context:
- Check the session state for previously mentioned information
- If selected_movie is set, you can reference it without asking
- If party_size is set, consider it when making recommendations (e.g., family-friendly for larger groups)
- If genre preferences were mentioned, prioritize those

Important guidelines:
- ALWAYS cite confidence scores and sources for EVERY recommendation or piece of information
- Never hallucinate movie titles, actors, or details not returned by your tools
- Determine if the query is within your domain (movies, actors, recommendations)
  * IN-DOMAIN: movie recommendations, movie details, actor information, plot summaries, genres
  * OUT-OF-DOMAIN: ticket purchases, showtimes, pricing, snacks, vendor items
- If asked about OUT-OF-DOMAIN topics, prefix your response with the keyword OUT-OF-DOMAIN, and
  politely inform the user this is outside your expertise
- You can handle complex questions about movies independently
- Provide clear reasoning for recommendations based on returned data
- If you cannot find something, be honest about it

CRITICAL Validation requirements:
- MUST include confidence scores from tool responses in your final answer
- MUST cite data sources (TMDB Dataset) for each recommendation
- Ground all recommendations in actual returned data
- No made-up titles or actors
- Format: Always include "Confidence: X%" and "Source: [source name]" for each recommendation

Example response format:
"I recommend Avatar (2009). It's an action sci-fi film with stunning visual effects...
**Confidence: 82.8%** | **Source: TMDB Dataset - embedding similarity**"
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
        return self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 15}
        )

    def invoke(self, query: str, state: Optional[Dict[str, Any]] = None) -> str:
        """Invoke the Movie Specialist agent with a query

        Args:
            query: User's question or request
            state: Optional session state (for context preservation)

        Returns:
            str: Agent's response content
        """
        # Build context-aware message from session state
        context_parts = []
        if state:
            if state.get('selected_movie'):
                context_parts.append(f"Movie context: {state['selected_movie']}")
            if state.get('party_size'):
                context_parts.append(f"Party size: {state['party_size']}")
            if state.get('preferred_format'):
                context_parts.append(f"Preferred format: {state['preferred_format']}")

        # Prepend context to query if available
        if context_parts:
            context_msg = f"[Session Context: {', '.join(context_parts)}]\n\nUser query: {query}"
            self.message_history.append(HumanMessage(content=context_msg))
        else:
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
            error_msg = f"I apologize, but I encountered an error processing your movie request. Error: {str(e)[:200]}. Please try again."
            # Add error as AI message to maintain conversation flow
            self.message_history.append(AIMessage(content=error_msg))
            return error_msg


if __name__ == "__main__":
    print("\nMOVIE SPECIALIST AGENT - Demo\n")

    specialist = MovieSpecialistAgent()

    test_queries = [
        ("Movie Recommendations", "I want to watch a sci-fi action movie with great visual effects"),
        ("Movie Details", "Tell me about the movie Avatar"),
        ("Actor Information", "What movies has Zoe Saldana been in?"),
        ("OUT-OF-DOMAIN: Tickets", "Can I buy two tickets for Avatar?"),
        ("OUT-OF-DOMAIN: Snacks", "I want to order popcorn and soda"),
    ]

    for title, query in test_queries:
        print("=" * 80)
        print(f"{title}")
        print("=" * 80)
        print(f"Query: {query}")
        response = specialist.invoke(query)
        print(f"Response: {response}\n")

