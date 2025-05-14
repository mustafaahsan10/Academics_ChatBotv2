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
    course_code: Optional[str] = Field(None, description="Course code mentioned in the query (e.g., 'CSC 226')")
    course_name: Optional[str] = Field(None, description="Course name mentioned in the query")
    keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
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
def extract_keywords(query: str) -> List[str]:
    """Extract keywords from the query"""
    stopwords = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "in", 
                 "on", "at", "to", "for", "with", "by", "about", "like", "how", 
                 "what", "when", "where", "why", "who", "which"]
    
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    return keywords

def extract_course_code(query: str) -> Optional[str]:
    """Extract course code from query"""
    course_pattern = r'\b([A-Z]{2,4})\s?(\d{3}[A-Z]?)\b'
    course_matches = re.findall(course_pattern, query)
    if course_matches:
        return f"{course_matches[0][0]} {course_matches[0][1]}"
    return None

def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
    """Perform hybrid search with Qdrant"""
    try:
        logger.info("Starting hybrid search...")
        
        # Get query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Extract potential filter terms
        keywords = extract_keywords(query)
        course_code = extract_course_code(query)
        
        # Prepare metadata filter
        metadata_filter = None
        if course_code:
            metadata_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.course_code",
                        match=qdrant_models.MatchValue(value=course_code)
                    )
                ]
            )
        elif "engineering" in query.lower():
            metadata_filter = qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.keywords",
                        match=qdrant_models.MatchAny(any=["engineering", "Faculty of Engineering", "FE"])
                    )
                ]
            )
        
        # Vector search with optional filter
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=metadata_filter,
            limit=top_k
        )
        
        logger.info(f"Found {len(search_results)} results in hybrid search")
        return search_results
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        # Fall back to basic vector search
        try:
            logger.info("Attempting fallback search...")
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
        return "No specific information found."
    
    context_parts = []
    for i, hit in enumerate(results):
        payload = hit.payload
        text = payload.get("text", "")
        metadata = payload.get("metadata", {})
        
        # Format section info
        section_info = ""
        if metadata.get("heading"):
            section_info = f"Section: {metadata['heading']}\n"
        
        # Format content
        context_parts.append(f"{section_info}{text}")
    
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
        # Perform hybrid search
        search_results = hybrid_search(request.query, top_k=5)
        
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
    
    if "model" in st.session_state:
        # Check if the model is from OpenRouter
        selected_model = st.session_state.model
        if selected_model.startswith("openrouter/"):
            # For OpenRouter models - remove the prefix for compatibility
            model_id = selected_model.replace("openrouter/", "")
        else:
            # For OpenAI models
            model_id = selected_model
    
    logger.info(f"Using model: {model_id} for course information with tools")
    
    # Create the agent with tools
    course_agent = pydantic_ai.Agent(
        model=model_id,
        tools=[search_course_info],
        system_prompt="""You are a knowledgeable university assistant specializing in course information.
Your goal is to provide accurate, helpful information about courses, their content, and related details.

You have access to a search_course_info tool that can find information about university courses, degree programs, 
and academic requirements. Always use this tool to look up information before answering questions.

When responding to queries about courses:
1. Be specific about course codes and official course names
2. Include important information found in the search results
3. Format your answers in a clear, structured way
4. If information is missing or uncertain, clearly state this

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your accuracy for their academic planning, so be specific and clear.
"""
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
        # Get the course agent with tools
        course_agent = get_course_agent()
        
        # Prepare user message with language preference
        user_message = query
        if language.lower() == "arabic":
            user_message = f"{query} (Please respond in Arabic)"
        
        # Call the agent with the query
        logger.info(f"Calling course agent with query: {query}, language: {language}")
        
        # Run the agent with the query - the agent will use the search_course_info tool as needed
        response = await course_agent.run(user_message)
        
        logger.info(f"Received response type: {type(response)}")
        
        # Return the response
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