import os
import json
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any
import pydantic_ai
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
import streamlit as st
import re
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "course_information"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Pydantic models for course information
class CourseQueryContext(BaseModel):
    """Context for a course information query"""
    user_query: str = Field(..., description="The user's original query about course information")
    processed_query: str = Field("", description="The processed query with stopwords removed")
    keywords_matches: List[Dict] = Field(default_factory=list, description="Matches found in the keywords field")
    heading_matches: List[Dict] = Field(default_factory=list, description="Matches found in the heading field")
    context: Optional[str] = Field(None, description="Context information retrieved from search")
    response_language: str = Field("English", description="Language to respond in (English or Arabic)")

class CourseSearchRequest(BaseModel):
    """Request parameters for searching course information"""
    query: str = Field(..., description="The query to search for course information")
    language: str = Field("English", description="The language to respond in")

class CourseSearchResult(BaseModel):
    """Result from searching course information"""
    context: str = Field(..., description="Context information retrieved from search")

class CourseResponse(BaseModel):
    """Structured response for course information queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query about courses")

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

def extract_terms_of_interest(query: str) -> List[str]:
    """Extract important academic terms from the query"""
    academic_terms = [
        "major", "majors", "program", "programs", "degree", "degrees", "faculty", "faculties",
        "undergraduate", "graduate", "credits", "credit", "course", "courses", "prerequisite",
        "prerequisites", "requirement", "requirements", "engineering", "business", "science",
        "arts", "humanities", "law", "nursing", "health", "computer", "architecture"
    ]
    
    # Look for these terms in the query
    found_terms = []
    for term in academic_terms:
        if term in query.lower().split():
            found_terms.append(term)
    
    return found_terms

def hierarchical_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Perform hierarchical search by first using vector search and then
    post-processing to prioritize results with matching keywords and headings
    """
    try:
        logger.info(f"Starting hierarchical search for query: {query}")
        
        # Process the query
        processed_query = process_query(query)
        terms_of_interest = extract_terms_of_interest(query)
        
        logger.info(f"Processed query: '{processed_query}', terms: {terms_of_interest}")
        
        # Get query embedding for vector search
        query_embedding = embeddings.embed_query(processed_query)
        
        # Get a larger initial result set for reranking
        initial_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=25  # Get more results for post-filtering
        )
        
        logger.info(f"Found {len(initial_results)} results in vector search")
        
        if not initial_results:
            logger.info("No results found")
            return []
        
        # Post-process results to implement hierarchical search
        # We'll examine the results and boost those with keyword/heading matches
        enhanced_results = []
        
        for result in initial_results:
            payload = result.payload
            metadata = payload.get("metadata", {})
            
            # Initialize score components
            base_score = result.score
            keyword_boost = 0.0
            heading_boost = 0.0
            
            # 1. Check for matches in keywords
            keywords = metadata.get("keywords", [])
            if isinstance(keywords, list) and keywords:
                # Look for matches between query terms and keywords
                for term in processed_query.split():
                    # Check if term appears in any keyword
                    if any(term.lower() in kw.lower() for kw in keywords):
                        keyword_boost += 0.1
                
                # Extra boost for terms of interest
                for term in terms_of_interest:
                    if any(term.lower() in kw.lower() for kw in keywords):
                        keyword_boost += 0.2
            
            # 2. Check for matches in heading
            heading = metadata.get("heading", "")
            if heading:
                # Look for matches between query terms and heading
                heading_lower = heading.lower()
                for term in processed_query.split():
                    if term.lower() in heading_lower:
                        heading_boost += 0.15
                
                # Extra boost for terms of interest in heading
                for term in terms_of_interest:
                    if term.lower() in heading_lower:
                        heading_boost += 0.25
            
            # 3. Calculate final score with boosts
            final_score = base_score * (1.0 + keyword_boost + heading_boost)
            
            # Create enhanced result with new score
            enhanced_result = {
                "id": result.id,
                "payload": payload,
                "score": final_score,
                "vector": result.vector if hasattr(result, "vector") else None,
                "boosts": {
                    "keyword_boost": keyword_boost,
                    "heading_boost": heading_boost
                }
            }
            
            enhanced_results.append(enhanced_result)
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k results
        top_results = enhanced_results[:top_k]
        
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
            fallback_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k
            )
            logger.info(f"Fallback search found {len(fallback_results)} results")
            return fallback_results
        except Exception as e2:
            logger.error(f"Fallback search failed: {e2}")
            return []

def format_search_results(results) -> str:
    """Format search results into a context string"""
    if not results:
        return "No specific information found related to your query."
    
    context_parts = []
    for i, hit in enumerate(results):
        payload = hit.payload
        text = payload.get("text", "")
        metadata = payload.get("metadata", {})
        
        # Format heading info
        heading_info = ""
        if metadata.get("heading"):
            heading_info = f"Section: {metadata['heading']}\n"
        
        # Format keywords for context
        keywords_info = ""
        if metadata.get("keywords") and isinstance(metadata.get("keywords"), list):
            keywords = metadata.get("keywords")
            keywords_info = f"Keywords: {', '.join(keywords)}\n"
        
        # Format content with score
        context_parts.append(f"{heading_info}{keywords_info}\nContent: {text}\n(Relevance Score: {hit.score:.2f})")
    
    return "\n\n---\n\n".join(context_parts)

# Create a tool for course information search
def search_course_info(request: CourseSearchRequest) -> CourseSearchResult:
    """
    Search for course information based on the query
    
    Args:
        request: The search request with query and language
        
    Returns:
        Results matching the search query
    """
    try:
        # Perform hierarchical search
        search_results = hierarchical_search(request.query, top_k=5)
        
        # Format the results
        context = format_search_results(search_results)
        
        return CourseSearchResult(
            context=context
        )
            
    except Exception as e:
        logger.error(f"Error searching course information: {e}")
        return CourseSearchResult(
            context="Error searching for course information."
        )

# Function to get an AI agent with the dynamically selected model
def get_course_agent():
    """
    Get the AI agent for course information with the model selected by the user
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
    
    logger.info(f"Using model: {model_id} for course information with tools")
    if use_openrouter:
        logger.info(f"Using OpenRouter API for model: {model_id}")
    
    # Create the agent with tools
    # Use the appropriate API configuration based on whether it's an OpenRouter model
    course_agent = pydantic_ai.Agent(
        model=model_id,
        tools=[search_course_info],
        system_prompt="""You are a knowledgeable university assistant specializing in course information.
    Your goal is to provide accurate, helpful information about courses, their content, programs, majors, 
    and academic requirements.

    You have access to a search_course_info tool that can find information about university courses, degree programs, 
    academic requirements, and available majors. Always use this tool to look up information before answering questions.

    When responding to queries:
    1. Be specific about program names and degree requirements
    2. Include important information found in the search results
    3. Format your answers in a clear, structured way
    4. If information is missing or uncertain, clearly state this

    For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

    Remember that students rely on your accuracy for their academic planning, so be specific and clear.
    """,
        openai_api_key=openrouter_api_key if use_openrouter else openai_api_key,
        openai_api_base="https://openrouter.ai/api/v1" if use_openrouter else None
    )
    return course_agent

async def get_course_response(query: str, language: str = "English") -> str:
    """
    Get a response for a course information query
    
    Args:
        query: The user's question about courses
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
            search_request = CourseSearchRequest(query=query, language=language)
            search_result = search_course_info(search_request)
            context = search_result.context
            
            # Prepare system message with context
            system_message = """You are a knowledgeable university assistant specializing in course information.
Your goal is to provide accurate, helpful information about courses, their content, programs, majors, 
and academic requirements based on the following information:

""" + context + """

When responding to queries:
1. Be specific about program names and degree requirements
2. Include important information found in the search results
3. Format your answers in a clear, structured way
4. If information is missing or uncertain, clearly state this

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
        course_agent = get_course_agent()
        response = await course_agent.run(user_message)
        logger.info(f"Received response type: {type(response)}")
        return response.output
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating course response: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة استفسارك عن المقرر الدراسي. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your course information query. Please try again."

def get_course_response_sync(query: str, language: str = "English") -> str:
    """
    Synchronous wrapper for get_course_response
    
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
        
    return loop.run_until_complete(get_course_response(query, language))