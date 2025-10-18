"""
Movie Specialist Agent

Handles movie recommendations, details, and actor information.
Max 3 tools enforced.
"""

import json
import os
from typing import Annotated, List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()


# Structured output schema
class MovieOutput(TypedDict):
    """Structured output from Movie Specialist Agent"""
    competent: bool  # If user's query is out of domain, return False. Otherwise return True
    content: str  # Natural language response with confidence and source


# Global data store (loaded once)
_movies_df = None
_credits_df = None
_movie_embeddings = None
_movie_id_to_idx = {}


def load_movie_data():
    """Load and prepare movie data with embeddings"""
    global _movies_df, _credits_df, _movie_embeddings, _movie_id_to_idx

    if _movies_df is not None:
        return  # Already loaded

    # Load datasets
    _movies_df = pd.read_csv('tmdb_5000_movies_small.csv', delimiter=';')
    _credits_df = pd.read_csv('tmdb_5000_credits_small.csv', delimiter=';')

    # Parse JSON columns
    _movies_df['genres_parsed'] = _movies_df['genres'].apply(
        lambda x: [g['name'] for g in json.loads(x)] if pd.notna(x) else []
    )
    _movies_df['keywords_parsed'] = _movies_df['keywords'].apply(
        lambda x: [k['name'] for k in json.loads(x)] if pd.notna(x) else []
    )

    # Parse cast from credits
    _credits_df['cast_parsed'] = _credits_df['cast'].apply(
        lambda x: json.loads(x) if pd.notna(x) else []
    )

    # Merge credits with movies
    _movies_df = _movies_df.merge(_credits_df[['movie_id', 'cast_parsed']],
                                   left_on='id', right_on='movie_id', how='left')

    # Create searchable text for embeddings
    _movies_df['search_text'] = _movies_df.apply(
        lambda row: f"{row['title']}. {' '.join(row['genres_parsed'])}. {row['overview']}. "
                   f"Keywords: {' '.join(row['keywords_parsed'][:10])}",
        axis=1
    )

    # Create movie ID to index mapping
    _movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(_movies_df['id'].tolist())}

    # Generate embeddings for movies (batch process to save API calls)
    # For simplicity, we'll generate embeddings on-demand or cache them
    print(f"Loaded {len(_movies_df)} movies")


# Tool 1: Movie Recommendation System (embedding-based)
@tool
def recommend_movies(
    query: Annotated[str, "User's movie preference or description (e.g., 'action sci-fi', 'romantic comedy')"],
    limit: Annotated[int, "Maximum number of recommendations to return"] = 5
) -> str:
    """
    Find movie recommendations based on semantic similarity to user preferences.
    Uses embeddings to match movies to user's described preferences.
    Returns ranked list with reasons and confidence scores.
    """
    load_movie_data()

    try:
        # Get embedding for user query
        query_embedding = embeddings.embed_query(query)

        # Get or create movie embeddings (cache for efficiency)
        # For demo, we'll compute a subset
        sample_size = min(50, len(_movies_df))
        sample_df = _movies_df.head(sample_size).copy()

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

        print("----- found movies")
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


# Tool 2: Movie Information Lookup
@tool
def get_movie_details(
    title: Annotated[str, "Movie title to look up"],
) -> str:
    """
    Get detailed information about a specific movie including plot, genres,
    ratings, runtime, and release date. Returns comprehensive movie metadata.
    """
    load_movie_data()

    try:
        # Case-insensitive search
        matches = _movies_df[_movies_df['title'].str.lower() == title.lower()]

        if matches.empty:
            # Try partial match
            matches = _movies_df[_movies_df['title'].str.contains(title, case=False, na=False)]

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

        print("------ found")
        return json.dumps(details, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "confidence": 0.0,
            "source": "TMDB Dataset"
        })


# Tool 3: Actor Information Lookup
@tool
def get_actor_info(
    actor_name: Annotated[str, "Actor's name to look up"],
) -> str:
    """
    Get information about an actor including their filmography highlights
    and notable roles. Returns actor details and top movies they appeared in.
    """
    load_movie_data()

    print("-----loading....")
    print(f"Columns available: {_movies_df.columns.tolist()}")
    print(f"Has cast_parsed: {'cast_parsed' in _movies_df.columns}")

    try:
        # Search across all cast lists
        actor_movies = []

        # Check if cast_parsed exists
        if 'cast_parsed' not in _movies_df.columns:
            return json.dumps({
                "error": "cast_parsed column not found in dataframe",
                "available_columns": _movies_df.columns.tolist(),
                "confidence": 0.0,
                "source": "TMDB Dataset"
            })

        print(f"Searching for actor: {actor_name}")
        print(f"Total movies to search: {len(_movies_df)}")

        for idx, row in _movies_df.iterrows():
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
            print("--- found no actor movie")
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
        print("----- found")
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


# System prompt for Movie Specialist
MOVIE_SPECIALIST_SYSTEM_PROMPT = """You are the Movie Specialist at a movie theater.

Your responsibilities:
- Provide movie recommendations based on user preferences (genre, mood, themes)
- Share detailed information about movies (plot, ratings, runtime, etc.)
- Provide actor information and their filmography highlights

You have access to exactly 3 tools:
1. recommend_movies: Find movies matching user preferences using semantic similarity
2. get_movie_details: Get detailed information about a specific movie
3. get_actor_info: Get actor details and their top films

Important guidelines:
- ALWAYS cite confidence scores and sources for EVERY recommendation or piece of information
- Never hallucinate movie titles, actors, or details not returned by your tools
- Determine if the query is within your domain (movies, actors, recommendations)
  * IN-DOMAIN: movie recommendations, movie details, actor information, plot summaries, genres
  * OUT-OF-DOMAIN: ticket purchases, showtimes, pricing, snacks, vendor items
- If asked about OUT-OF-DOMAIN topics, politely inform the user this is outside your expertise
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


def create_movie_specialist_agent():
    """Create the Movie Specialist agent with exactly 3 tools"""
    tools = [recommend_movies, get_movie_details, get_actor_info]

    # Verify tool limit
    assert len(tools) <= 3, "Movie Specialist cannot have more than 3 tools"

    # Create agent with system prompt
    agent = create_react_agent(
        llm,
        tools,
        prompt=(MOVIE_SPECIALIST_SYSTEM_PROMPT)
    )

    return agent


# Helper function to invoke the agent with structured output
def invoke_movie_specialist(query: str, state: Optional[Dict[str, Any]] = None) -> MovieOutput:
    """
    Invoke the Movie Specialist agent with a query

    Args:
        query: User's question or request
        state: Optional session state (for context preservation)

    Returns:
        MovieOutput with competent flag and content
    """
    agent = create_movie_specialist_agent()

    # Prepare messages
    messages = [HumanMessage(content=query)]

    # Invoke agent
    result = agent.invoke({"messages": messages})

    # Extract response content
    agent_response = result["messages"][-1].content if result["messages"] else "No response"

    # Determine competency based on response and query
    # Check if the response indicates out-of-domain
    out_of_domain_keywords = [
        "outside my expertise", "not my area", "ticket master", "vendor",
        "can't help with tickets", "can't help with snacks", "pricing",
        "showtimes", "purchase tickets", "buy snacks"
    ]

    competent = not any(keyword.lower() in agent_response.lower() for keyword in out_of_domain_keywords)

    # Also check query for obvious out-of-domain requests
    query_lower = query.lower()
    if any(word in query_lower for word in ["ticket", "showtime", "snack", "popcorn", "soda", "price", "buy", "purchase"]):
        # These might be out of domain, but let the agent's response determine
        competent = False

    # Return structured output
    return MovieOutput(
        competent=competent,
        content=agent_response
    )


if __name__ == "__main__":
    # Test the agent
    print("Testing Movie Specialist Agent with Structured Output...\n")

    # Test 1: In-domain query - Movie Recommendations
    print("=" * 80)
    print("Test 1: Movie Recommendations (IN-DOMAIN)")
    print("=" * 80)
    query1 = "I want to watch a sci-fi action movie with great visual effects"
    response1 = invoke_movie_specialist(query1)
    print(f"Query: {query1}")
    print(f"Competent: {response1['competent']}")
    print(f"Response: {response1['content']}\n")

    # Test 2: In-domain query - Movie Details
    print("=" * 80)
    print("Test 2: Movie Details (IN-DOMAIN)")
    print("=" * 80)
    query2 = "Tell me about the movie Avatar"
    response2 = invoke_movie_specialist(query2)
    print(f"Query: {query2}")
    print(f"Competent: {response2['competent']}")
    print(f"Response: {response2['content']}\n")

    # Test 3: In-domain query - Actor Info
    print("=" * 80)
    print("Test 3: Actor Information (IN-DOMAIN)")
    print("=" * 80)
    query3 = "What movies has Zoe Saldana been in?"
    response3 = invoke_movie_specialist(query3)
    print(f"Query: {query3}")
    print(f"Competent: {response3['competent']}")
    print(f"Response: {response3['content']}\n")

    # Test 4: Out-of-domain query - Tickets
    print("=" * 80)
    print("Test 4: Ticket Purchase (OUT-OF-DOMAIN)")
    print("=" * 80)
    query4 = "Can I buy two tickets for Avatar?"
    response4 = invoke_movie_specialist(query4)
    print(f"Query: {query4}")
    print(f"Competent: {response4['competent']}")
    print(f"Response: {response4['content']}\n")

    # Test 5: Out-of-domain query - Snacks
    print("=" * 80)
    print("Test 5: Snack Ordering (OUT-OF-DOMAIN)")
    print("=" * 80)
    query5 = "I want to order popcorn and soda"
    response5 = invoke_movie_specialist(query5)
    print(f"Query: {query5}")
    print(f"Competent: {response5['competent']}")
    print(f"Response: {response5['content']}\n")

