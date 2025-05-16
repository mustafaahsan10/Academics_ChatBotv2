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
COLLECTION_NAME = "exam_alerts"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# class ExamQueryContext(BaseModel):
#     """Context for an exam/deadline query"""
#     user_query: str = Field(..., description="The user's original query about exams or deadlines")
#     course_code: Optional[str] = Field(None, description="Course code mentioned in the query (e.g., 'CS101')")
#     exam_type: Optional[str] = Field(None, description="Type of exam mentioned (e.g., 'midterm', 'final', 'quiz')")
#     date_mentioned: Optional[str] = Field(None, description="Date or time period mentioned (e.g., 'next week', 'December')")
#     location_mentioned: Optional[str] = Field(None, description="Location mentioned for the exam")
#     assessment_type: Optional[str] = Field(None, description="Type of assessment (e.g., 'assignment', 'project', 'presentation')")
#     professor_name: Optional[str] = Field(None, description="Professor name mentioned in the query")
#     is_upcoming_request: bool = Field(False, description="Whether the query is about upcoming exams or deadlines")
#     keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
#     search_results: Optional[List[Dict[Any, Any]]] = Field(None, description="Relevant exam information retrieved")
#     response_language: str = Field("English", description="Language to respond in (English or Arabic)")

class ExamSearchRequest(BaseModel):
    """Request parameters for searching exam information"""
    query: str = Field(..., description="The query to search for exam information")
    language: str = Field("English", description="The language to respond in")

class ExamSearchResult(BaseModel):
    """Result from searching exam information"""
    context: str = Field(..., description="Context information retrieved from search")

class ExamResponse(BaseModel):
    """Structured response for exam/deadline queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query about exams or deadlines")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions related to exams")

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

def extract_exam_entities(query: str) -> Dict[str, Any]:
    """Extract exam-related entities from the query"""
    entities = {
        "course_code": None,
        "exam_type": None,
        "date_mentioned": None,
        "location_mentioned": None,
        "is_upcoming_request": False,
        "keywords": []
    }
    
    # Look for course codes (e.g., CSC 226, MTH110)
    course_code_pattern = r'\b([A-Z]{2,4})\s*(\d{3}[A-Z0-9]*)\b'
    course_code_matches = re.findall(course_code_pattern, query, re.IGNORECASE)
    if course_code_matches:
        # Format as "CSC 226"
        entities["course_code"] = f"{course_code_matches[0][0].upper()} {course_code_matches[0][1]}"
        entities["keywords"].append(entities["course_code"])
    
    # Look for exam types
    exam_types = ["final", "midterm", "quiz", "test", "exam", "assessment"]
    for exam_type in exam_types:
        if exam_type in query.lower():
            entities["exam_type"] = exam_type
            entities["keywords"].append(exam_type)
            break
    
    # Look for date mentions
    date_keywords = ["tomorrow", "today", "next week", "next month", "upcoming", "future", "schedule", "calendar"]
    for date_keyword in date_keywords:
        if date_keyword in query.lower():
            entities["date_mentioned"] = date_keyword
            entities["keywords"].append(date_keyword)
            # Check if this is an upcoming request
            if date_keyword in ["tomorrow", "next week", "next month", "upcoming", "future"]:
                entities["is_upcoming_request"] = True
            break
    
    # Look for location mentions
    location_keywords = ["where", "location", "room", "building", "hall", "campus", "online"]
    for location_keyword in location_keywords:
        if location_keyword in query.lower():
            entities["location_mentioned"] = location_keyword
            entities["keywords"].append(location_keyword)
            break
    
    # Check for general upcoming request terms
    upcoming_terms = ["when", "upcoming", "future", "schedule", "released", "published"]
    if any(term in query.lower() for term in upcoming_terms):
        entities["is_upcoming_request"] = True
    
    # Add exam-related keywords
    exam_keywords = ["schedule", "date", "time", "deadline", "exam", "final", "midterm", "assessment"]
    for keyword in exam_keywords:
        if keyword in query.lower() and keyword not in entities["keywords"]:
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
        entities = extract_exam_entities(query)
        
        # Get query embedding for vector search
        query_embedding = embeddings.embed_query(processed_query)
        
        # Get a larger initial result set for reranking
        initial_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=100  # Increased to get more potential matches
        )
        
        logger.info(f"Found {len(initial_results)} results in vector search")
        
        if not initial_results:
            logger.info("No results found")
            return []
        
        # Post-process results to implement hierarchical search
        enhanced_results = []
        
        for result in initial_results:
            payload = result.payload
            metadata = payload.get("metadata", {})
            
            # Initialize score components
            base_score = result.score
            course_code_boost = 0.0
            exam_type_boost = 0.0
            date_boost = 0.0
            location_boost = 0.0
            keyword_boost = 0.0
            
            # 1. Check for course code match
            if entities["course_code"] and "course_code" in payload:
                if entities["course_code"].lower() == payload["course_code"].lower():
                    course_code_boost = 0.5  # Strong boost for exact course code match
                elif entities["course_code"].split()[0].lower() in payload["course_code"].lower():
                    course_code_boost = 0.2  # Partial match (just the prefix)
            
            # 2. Check for exam type match
            if entities["exam_type"] and "type" in payload:
                if entities["exam_type"].lower() in payload["type"].lower():
                    exam_type_boost = 0.4
            
            # 3. Check for date mentions
            if entities["date_mentioned"] and "date" in payload:
                # Simple date check - could be enhanced with date parsing logic
                if entities["date_mentioned"] in payload["date"]:
                    date_boost = 0.3
            
            # 4. Check for location match
            if entities["location_mentioned"] and "location" in payload:
                if entities["location_mentioned"].lower() in payload["location"].lower():
                    location_boost = 0.3
            
            # 5. Check for general keyword matches
            if "keywords" in payload and isinstance(payload["keywords"], list):
                for keyword in entities["keywords"]:
                    if any(keyword.lower() in kw.lower() for kw in payload["keywords"]):
                        keyword_boost += 0.05
            
            # Calculate final score with boosts
            final_score = base_score * (1.0 + course_code_boost + exam_type_boost + 
                                        date_boost + location_boost + keyword_boost)
            
            # Create enhanced result with new score
            enhanced_result = {
                "id": result.id,
                "payload": payload,
                "score": final_score,
                "boosts": {
                    "course_code_boost": course_code_boost,
                    "exam_type_boost": exam_type_boost,
                    "date_boost": date_boost,
                    "location_boost": location_boost,
                    "keyword_boost": keyword_boost
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

def format_exam_results(results) -> str:
    """Format search results into a context string"""
    if not results:
        return "No specific information found related to your query about exams or deadlines."
    
    context_parts = []
    for i, hit in enumerate(results):
        payload = hit.payload
        
        # Format the exam information in a readable way
        exam_info = f"Course: {payload.get('course_code', 'N/A')} - {payload.get('course_name', 'Unknown')}\n"
        exam_info += f"Type: {payload.get('type', 'Final Exam')}\n"
        exam_info += f"Date: {payload.get('date', 'Not specified')}\n"
        exam_info += f"Time: {payload.get('time', 'Not specified')}\n"
        exam_info += f"Location: {payload.get('location', 'Not specified')}\n"
        
        # Add keywords if available
        if payload.get("keywords") and isinstance(payload.get("keywords"), list):
            keywords = payload.get("keywords")
            exam_info += f"Keywords: {', '.join(keywords)}\n"
        
        # Add detailed text if available
        if payload.get("text"):
            exam_info += f"\nDetails: {payload.get('text')}\n"
            
        # Add relevance score
        exam_info += f"(Relevance Score: {hit.score:.2f})"
        
        context_parts.append(exam_info)
    
    return "\n\n---\n\n".join(context_parts)

# Create a tool for exam information search
def search_exam_info(request: ExamSearchRequest) -> ExamSearchResult:
    """
    Search for exam information based on the query
    
    Args:
        request: The search request with query and language
        
    Returns:
        Results matching the search query
    """
    try:
        # Perform hierarchical search - return all results
        search_results = hierarchical_search(request.query, top_k=None)
        
        # Format the results
        context = format_exam_results(search_results)
        
        return ExamSearchResult(
            context=context
        )
            
    except Exception as e:
        logger.error(f"Error searching exam information: {e}")
        return ExamSearchResult(
            context="Error searching for exam information."
        )

def get_exam_agent():
    """
    Get the AI agent for exam alerts with the model selected by the user
    """
    # Get model from Streamlit session state if available, otherwise use default
    model = "gpt-4o-mini"  # Default model
    use_openrouter = False
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if "model" in st.session_state:
        # Get the model from session state
        model_id = st.session_state.model
        
        # Check if we should use OpenRouter
        if "use_openrouter" in st.session_state:
            use_openrouter = st.session_state.use_openrouter
    
    logger.info(f"Using model: {model} for exam alerts with tools")
    
    # Create the agent with tools
    system_prompt = """You are a knowledgeable university assistant specializing in exam schedules and academic deadlines.
Your goal is to provide accurate, timely information about exams, assignments, and important academic dates.

You have access to a search_exam_info tool that can find information about exam schedules, dates, times, locations, and deadlines.
Always use this tool to look up information before answering questions about exams.

When responding to queries about exams and deadlines:
1. Prioritize upcoming events that are most relevant to the student
2. Include specific dates, times, and locations when available
3. Mention any special requirements (e.g., materials allowed, submission format)
4. Note the weight or importance of assessments when relevant

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about university exams and deadlines. Never say phrases like "I don't have information on this" or "this isn't in my database". Always maintain a helpful tone and try to address the user's query as best as possible.

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your accuracy for their academic planning, so be specific and clear.
"""
    
    # Create the agent with tools and specific API configuration
    if use_openrouter:
        logger.info(f"Using OpenRouter API for model: {model}")
        exam_agent = pydantic_ai.Agent(
            model,
            tools=[search_exam_info],
            input_type=ExamSearchRequest,
            output_type=ExamResponse,
            system_prompt=system_prompt,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        # Use regular OpenAI API
        exam_agent = pydantic_ai.Agent(
            model,
            tools=[search_exam_info],
            input_type=ExamSearchRequest,
            output_type=ExamResponse,
            system_prompt=system_prompt,
            openai_api_key=openai_api_key
        )
    
    return exam_agent

async def get_exam_response(query: str, language: str = "English") -> str:
    """
    Get a response for an exam or deadline query
    
    Args:
        query: The user's question about exams or deadlines
        language: The language to respond in
        
    Returns:
        Formatted response string
    """
    try:
        # Prepare user message with language preference
        user_message = query
        if language.lower() == "arabic":
            user_message = f"{query} (Please respond in Arabic)"
        
        # Get model information from session state
        use_openrouter = False
        model_id = "gpt-4o-mini"  # Default model
        
        if "model" in st.session_state:
            model_id = st.session_state.model
            if "use_openrouter" in st.session_state:
                use_openrouter = st.session_state.use_openrouter
                
        logger.info(f"Processing query with model: {model_id}, using OpenRouter: {use_openrouter}")
        
        # Get the exam agent
        exam_agent = get_exam_agent()
        
        # Create search request - extract query as string
        search_request_query = query
        
        # Run the agent and get the response
        if use_openrouter:
            # For OpenRouter, we may need a different approach
            import requests
            import json
            
            # Get search results first
            search_result = search_exam_info(search_request_query)
            context = search_result.context
            
            # Prepare system message with context
            system_message = """You are a knowledgeable university assistant specializing in exam schedules and academic deadlines.
Your goal is to provide accurate, timely information about exams, assignments, and important academic dates
based on the following information:

""" + context + """

When responding to queries about exams and deadlines:
1. Prioritize upcoming events that are most relevant to the student
2. Include specific dates, times, and locations when available

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about university exams and deadlines. Never say phrases like "I don't have information on this" or "this isn't in my database". Always maintain a helpful tone and try to address the user's query as best as possible.

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.
"""
            
            # Make direct API call to OpenRouter
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://university-chatbot.com",  # Replace with your actual site URL
                "X-Title": "University Chatbot"  # Replace with your actual site name
            }
            
            # Format the model name for OpenRouter
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
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            
            # Check if request was successful
            if response.status_code == 200:
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    return response_data["choices"][0]["message"]["content"]
                else:
                    raise ValueError("Invalid response format from OpenRouter")
            else:
                raise ValueError(f"OpenRouter API error: {response.status_code}")
        else:
            # Standard approach using pydantic_ai Agent
            response = await exam_agent.run(search_request_query)  # Pass string directly
            
            # Extract the answer from the response
            if hasattr(response, 'output') and hasattr(response.output, 'answer'):
                return response.output.answer
            else:
                return "Sorry, I couldn't process your query about exams. Please try again."
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating exam response: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة استفسارك عن الاختبارات والمواعيد النهائية. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your exam query. Please try again."

def get_exam_response_sync(query: str, language: str = "English") -> str:
    """
    Synchronous wrapper for get_exam_response
    
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
        
    return loop.run_until_complete(get_exam_response(query, language))