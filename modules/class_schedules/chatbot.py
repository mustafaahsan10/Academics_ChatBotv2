import os
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any
import pydantic_ai
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import re
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "class_schedules"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Simplified context model for class schedules queries
# class ScheduleQueryContext(BaseModel):
#     """Context for a class schedule query"""
#     user_query: str = Field(..., description="The user's original query about class schedules")
#     course_code: Optional[str] = Field(None, description="Course code mentioned in the query (e.g., 'CSC 226')")
#     course_name: Optional[str] = Field(None, description="Course name mentioned (e.g., 'Database Systems')")
#     day: Optional[str] = Field(None, description="Day of the week mentioned (e.g., 'Monday', 'Tuesday')")
#     time: Optional[str] = Field(None, description="Time mentioned in the query")
#     keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
#     search_results: Optional[List[Dict[Any, Any]]] = Field(None, description="Relevant class schedule information retrieved")
#     response_language: str = Field("English", description="Language to respond in (English or Arabic)")

# Class Schedule search request and response models
class ScheduleSearchRequest(BaseModel):
    """Request parameters for searching class schedules"""
    query: str = Field(..., description="The query to search for class schedule information")
    language: str = Field("English", description="The language to respond in")

class ScheduleSearchResult(BaseModel):
    """Result from searching class schedule information"""
    context: str = Field(..., description="Context information retrieved from search")

# Response model for class schedules
class ScheduleResponse(BaseModel):
    """Structured response for class schedule queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query about class schedules")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions related to schedules")

# Initialize embeddings and Qdrant client
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Helper functions for search
def process_query(query: str) -> str:
    """Process the query to remove stopwords and prepare for search"""
    stopwords = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "in", 
                 "on", "at", "to", "for", "with", "by", "about", "like", "how", 
                 "what", "when", "where", "why", "who", "which", "can", "you", "tell",
                 "me", "list", "all", "available", "have", "does"]
    
    # Convert to lowercase and tokenize
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Remove stopwords and short words
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Join back into a string
    processed_query = " ".join(keywords)
    
    logger.info(f"Processed query: '{processed_query}' from original: '{query}'")
    return processed_query

def extract_schedule_entities(query: str) -> Dict[str, Any]:
    """Extract schedule-related entities from the query"""
    entities = {
        "course_code": None,
        "course_name": None,
        "day": None,
        "time": None,
        "keywords": []
    }
    
    # Look for course codes (e.g., CSC 226, MTH110)
    course_code_pattern = r'\b([A-Z]{2,4})\s*(\d{3}[A-Z0-9]*)\b'
    course_code_matches = re.findall(course_code_pattern, query, re.IGNORECASE)
    if course_code_matches:
        # Format as "CSC 226"
        entities["course_code"] = f"{course_code_matches[0][0].upper()} {course_code_matches[0][1]}"
    
    # Look for days of the week
    days = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    for day in days:
        if day in query.lower():
            entities["day"] = day.capitalize()
            entities["keywords"].append(day)
            break
    
    # Look for time mentions (simple pattern)
    time_pattern = r'\b(\d{1,2}):?(\d{2})?\s*(am|pm|AM|PM)?\b'
    time_matches = re.findall(time_pattern, query)
    if time_matches:
        entities["time"] = time_matches[0][0]
        entities["keywords"].append(f"time:{time_matches[0][0]}")
    
    # Add schedule related keywords
    schedule_keywords = ["schedule", "timetable", "class", "lecture", "time", "day", "meet", "session"]
    query_words = query.lower().split()
    for keyword in schedule_keywords:
        if keyword in query_words:
            entities["keywords"].append(keyword)
    
    logger.info(f"Extracted entities: {entities}")
    return entities

def hierarchical_search(query: str, top_k: int = None) -> List[Dict]:
    """
    Perform hierarchical search by first using vector search and then
    post-processing to prioritize results with matching metadata
    
    Args:
        query: The search query
        top_k: Maximum number of results to return (None = return all results)
        
    Returns:
        List of search results with enhanced scoring
    """
    try:
        logger.info(f"Starting hierarchical search for query: {query}")
        
        # Process the query
        processed_query = process_query(query)
        entities = extract_schedule_entities(query)
        
        # Get query embedding for vector search
        query_embedding = embeddings.embed_query(processed_query)
        
        # Get a larger initial result set for reranking
        initial_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=100  # Increased from 25 to get more potential matches
        )
        
        logger.info(f"Found {len(initial_results)} results in vector search")
        
        if not initial_results:
            logger.info("No results found")
            return []
        
        # Post-process results to implement hierarchical search
        enhanced_results = []
        
        for result in initial_results:
            payload = result.payload
            
            # Initialize score components
            base_score = result.score
            course_code_boost = 0.0
            course_name_boost = 0.0
            day_boost = 0.0
            time_boost = 0.0
            keyword_boost = 0.0
            heading_boost = 0.0
            
            # 1. Check for course code match
            if entities["course_code"] and "course_code" in payload:
                if entities["course_code"].lower() == payload["course_code"].lower():
                    course_code_boost = 0.5  # Strong boost for exact course code match
                elif entities["course_code"].split()[0].lower() in payload["course_code"].lower():
                    course_code_boost = 0.2  # Partial match (just the prefix)
            
            # 2. Check for course name match
            if "course_name" in payload:
                if entities["course_name"] and entities["course_name"].lower() in payload["course_name"].lower():
                    course_name_boost = 0.4
                # Also check for keywords in course name
                for keyword in processed_query.split():
                    if keyword.lower() in payload["course_name"].lower():
                        course_name_boost += 0.1
            
            # 3. Check for day match in sessions
            if entities["day"] and "sessions" in payload:
                for session in payload["sessions"]:
                    if entities["day"].lower() == session.get("day", "").lower():
                        day_boost = 0.3
                        break
            
            # 4. Check for time match in sessions
            if entities["time"] and "sessions" in payload:
                for session in payload["sessions"]:
                    if entities["time"] in session.get("time_slot", ""):
                        time_boost = 0.3
                        break
            
            # 5. Check for matches in keywords
            if "keywords" in payload and isinstance(payload["keywords"], list):
                for keyword in processed_query.split():
                    if any(keyword.lower() in kw.lower() for kw in payload["keywords"]):
                        keyword_boost += 0.05
            
            # 6. Check for matches in heading
            if "heading" in payload:
                for term in processed_query.split():
                    if term.lower() in payload["heading"].lower():
                        heading_boost += 0.15
            
            # Calculate final score with boosts
            final_score = base_score * (1.0 + course_code_boost + course_name_boost + 
                                       day_boost + time_boost + keyword_boost + heading_boost)
            
            # Create enhanced result with new score
            enhanced_result = {
                "id": result.id,
                "payload": payload,
                "score": final_score,
                "boosts": {
                    "course_code_boost": course_code_boost,
                    "course_name_boost": course_name_boost,
                    "day_boost": day_boost,
                    "time_boost": time_boost,
                    "keyword_boost": keyword_boost,
                    "heading_boost": heading_boost
                }
            }
            
            enhanced_results.append(enhanced_result)
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k results or all results if top_k is None
        if top_k is not None:
            top_results = enhanced_results[:top_k]
        else:
            top_results = enhanced_results
        
        # Convert back to format expected by the rest of the code
        final_results = []
        for result in top_results:
            # Create a ScoredPoint-like object
            class ScoredPoint:
                def __init__(self, id, payload, score):
                    self.id = id
                    self.payload = payload
                    self.score = score
            
            scored_point = ScoredPoint(
                id=result["id"],
                payload=result["payload"],
                score=result["score"]
            )
            
            final_results.append(scored_point)
        
        logger.info(f"Returned {len(final_results)} results after hierarchical boost")
        return final_results
        
    except Exception as e:
        logger.error(f"Error in hierarchical search: {e}")
        # Fall back to basic vector search
        try:
            logger.info("Attempting fallback vector search...")
            limit = 100 if top_k is None else top_k
            fallback_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit
            )
            logger.info(f"Fallback search found {len(fallback_results)} results")
            return fallback_results
        except Exception as e2:
            logger.error(f"Fallback search failed: {e2}")
            return []

def format_schedule_results(results) -> str:
    """Format search results into a context string"""
    if not results:
        return "No specific information found related to your query."
    
    context_parts = []
    for i, hit in enumerate(results):
        payload = hit.payload
        
        # Format the schedule information in a readable way
        schedule_info = f"Course: {payload.get('course_code', 'N/A')} - {payload.get('course_name', 'Unknown')}\n"
        
        if payload.get("heading"):
            schedule_info += f"Section: {payload.get('heading')}\n"
        
        if payload.get("program"):
            schedule_info += f"Program: {payload.get('program')}\n"
        
        # Format sessions
        if payload.get("sessions"):
            schedule_info += "Schedule:\n"
            for session in payload.get("sessions", []):
                day = session.get("day", "Unknown")
                time = session.get("time_slot", "Unknown time")
                schedule_info += f"- {day}: {time}\n"
        
        # Add keywords if available
        if payload.get("keywords") and isinstance(payload.get("keywords"), list):
            keywords = payload.get("keywords")
            schedule_info += f"Keywords: {', '.join(keywords)}\n"
        
        # Add detailed text if available
        if payload.get("text"):
            schedule_info += f"\nDetails: {payload.get('text')}\n"
            
        # Add relevance score
        schedule_info += f"(Relevance Score: {hit.score:.2f})"
        
        context_parts.append(schedule_info)
    
    return "\n\n---\n\n".join(context_parts)

# Create a tool for schedule information search
def search_schedule_info(request: ScheduleSearchRequest) -> ScheduleSearchResult:
    """
    Search for class schedule information based on the query
    
    Args:
        request: The search request with query and language
        
    Returns:
        Results matching the search query
    """
    try:
        # Perform hierarchical search - return all results
        search_results = hierarchical_search(request.query, top_k=None)
        
        # Format the results
        context = format_schedule_results(search_results)
        
        return ScheduleSearchResult(
            context=context
        )
            
    except Exception as e:
        logger.error(f"Error searching schedule information: {e}")
        return ScheduleSearchResult(
            context="Error searching for schedule information."
        )

# Function to get an AI agent with the dynamically selected model
def get_schedule_agent():
    """
    Get the AI agent for class schedules with the model selected by the user
    """
    # Get model from Streamlit session state if available, otherwise use default
    model_id = "gpt-4o-mini"  # Default model
    use_openrouter = False
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if "model" in st.session_state:
        model_id = st.session_state.model
        # Check if we should use OpenRouter
        if "use_openrouter" in st.session_state and st.session_state.use_openrouter:
            use_openrouter = True
    
    logger.info(f"Using model: {model_id} for class schedules with tools")
    if use_openrouter:
        logger.info(f"Using OpenRouter API for model: {model_id}")
    
    # Create the agent with tools
    schedule_agent = pydantic_ai.Agent(
        model=model_id,
        tools=[search_schedule_info],
        system_prompt="""You are a knowledgeable university assistant specializing in class schedules.
Your goal is to provide accurate, helpful information about class timings, locations, and related details.

You have access to a search_schedule_info tool that can find information about when and where classes meet, 
semester start and end dates, final exam schedules, and other schedule-related information.
Always use this tool to look up information before answering questions.

When responding to queries about class schedules:
1. Be precise about days and times
2. When referencing specific courses, include the course code and name
3. Include all relevant information found in the search results

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about university class schedules. Never say phrases like "I don't have information on this" or "this isn't in my database". Always maintain a helpful tone and try to address the user's query as best as possible.

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your accuracy for their attendance, so be as specific and clear as possible.
""",
        openai_api_key=openrouter_api_key if use_openrouter else openai_api_key,
        openai_api_base="https://openrouter.ai/api/v1" if use_openrouter else None
    )
    return schedule_agent

async def get_schedule_response(query: str, language: str = "English") -> str:
    """
    Get a response for a class schedule query
    
    Args:
        query: The user's question about class schedules
        language: The language to respond in
        
    Returns:
        Formatted response string
    """
    try:
        # Check if OpenRouter should be used
        use_openrouter = False
        model_id = "gpt-4o-mini"  # Default model
        
        if "model" in st.session_state:
            model_id = st.session_state.model
            if "use_openrouter" in st.session_state:
                use_openrouter = st.session_state.use_openrouter
                
        logger.info(f"Processing query with model: {model_id}, using OpenRouter: {use_openrouter}")
        
        # Prepare user message with language preference
        user_message = query
        if language.lower() == "arabic":
            user_message = f"{query} (Please respond in Arabic)"
        
        if use_openrouter:
            # Use direct OpenRouter API call
            import requests
            import json
            
            # Get the search results first
            search_request = ScheduleSearchRequest(query=query, language=language)
            search_result = search_schedule_info(search_request)
            context = search_result.context
            
            # Prepare system message with context
            system_message = """You are a knowledgeable university assistant specializing in class schedules.
Your goal is to provide accurate, helpful information about class timings, locations, and related details
based on the following information:

""" + context + """

When responding to queries about class schedules:
1. Be precise about days and times
2. When referencing specific courses, include the course code and name
3. Include all relevant information found in the search results

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about university class schedules. Never say phrases like "I don't have information on this" or "this isn't in my database". Always maintain a helpful tone and try to address the user's query as best as possible.

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.
"""
            
            # Get OpenRouter API key
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                logger.error("OPENROUTER_API_KEY not found in environment variables")
                raise ValueError("OpenRouter API key not found")
            
            # Make direct API call to OpenRouter
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://university-chatbot.com",  # Replace with your actual site URL
                "X-Title": "University Chatbot"  # Replace with your actual site name
            }
            
            # Format the model name for OpenRouter
            # Claude models need "anthropic/" prefix, and Gemini needs "google/" prefix
            if model_id == "claude-3-haiku" or model_id == "claude-3-sonnet" or model_id == "claude-3-opus":
                openrouter_model = f"anthropic/{model_id}"
            elif model_id == "gemini-2.0-flash-001":
                openrouter_model = f"google/{model_id}"
            else:
                openrouter_model = model_id
                
            logger.info(f"Using OpenRouter model: {openrouter_model}")
            
            # Prepare payload
            payload = {
                "model": openrouter_model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            }
            
            # Make the API request
            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(payload))
                
                # Check if request was successful
                if response.status_code == 200:
                    response_data = response.json()
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Unexpected response format: {response_data}")
                        raise ValueError("Invalid response format from OpenRouter")
                else:
                    logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                    raise ValueError(f"OpenRouter API error: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error making OpenRouter API call: {e}")
                # Fall back to using pydantic_ai agent with OpenAI model
                logger.info("Falling back to standard OpenAI model")
                # Clear the OpenRouter flag to use standard OpenAI
                use_openrouter = False
                model_id = "gpt-4o-mini"  # Use a reliable fallback model
        
        # Standard approach using pydantic_ai Agent
        logger.info(f"Using pydantic_ai agent with model: {model_id}")
        schedule_agent = get_schedule_agent()
        response = await schedule_agent.run(user_message)
        
        # Extract answer from response based on structure
        if hasattr(response, 'output'):
            if hasattr(response.output, 'answer'):
                return response.output.answer
            else:
                return str(response.output)
        else:
            return str(response)
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating schedule response: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة استفسارك عن جدول المحاضرات. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your class schedule query. Please try again."

def get_schedule_response_sync(query: str, language: str = "English") -> str:
    """
    Synchronous wrapper for get_schedule_response
    
    Args:
        query: The user's question
        language: The language to respond in
        
    Returns:
        Formatted response string
    """
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    return loop.run_until_complete(get_schedule_response(query, language))