import os
import json
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any
import pydantic_ai
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from pathlib import Path
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "course_information"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

class CourseQueryContext(BaseModel):
    """Simplified context for a course information query"""
    user_query: str = Field(..., description="The user's original query about course information")
    course_code: Optional[str] = Field(None, description="Course code mentioned in the query (e.g., 'CSC 226')")
    course_name: Optional[str] = Field(None, description="Course name mentioned in the query")
    keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
    search_results: Optional[List[Dict[Any, Any]]] = Field(None, description="Relevant course information retrieved")
    response_language: str = Field("English", description="Language to respond in (English or Arabic)")

class CourseResponse(BaseModel):
    """Structured response for course information queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query about courses")

# Function to get an AI agent with the dynamically selected model
def get_course_agent():
    """
    Get the AI agent for course information with the model selected by the user
    """
    # Get model from Streamlit session state if available, otherwise use default
    model = "openai:gpt-4o-mini"  # Default model
    
    if "model" in st.session_state:
        # Check if the model is from OpenRouter
        selected_model = st.session_state.model
        if selected_model.startswith("openrouter/"):
            # For OpenRouter models
            model = f"openrouter:{selected_model}"
        else:
            # For OpenAI models
            model = f"openai:{selected_model}"
            print("Used Model: ", model)
    
    logger.info(f"Using model: {model} for course information")
    
    return pydantic_ai.Agent(
        model,
        input_type=CourseQueryContext,
        output_type=CourseResponse,
        system_prompt="""You are a knowledgeable university assistant specializing in course information.
Your goal is to provide accurate, helpful information about courses, their content, and related details.

When responding to queries about courses:
1. Be specific about course codes and official course names
2. Include important information found in the search results
3. Format your answers in a clear, structured way
4. If information is missing or uncertain, clearly state this

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your accuracy for their academic planning, so be specific and clear.
"""
    )

async def add_course_search_context(context: CourseQueryContext) -> CourseQueryContext:
    """
    Add relevant search results to the context for course information queries
    
    Args:
        context: The current query context
        
    Returns:
        Updated context with search results
    """
    try:
        # Load course data from file
        data_path = Path("data/processed/admission_guide_PDF_extracted_text.json")
        if not data_path.exists():
            logger.warning(f"Courses data file not found at {data_path}")
            return context
            
        with open(data_path, "r", encoding="utf-8") as f:
            courses = json.load(f)
        
        # Convert to list if not already
        if isinstance(courses, dict):
            courses = [courses]
        
        # Apply filters based on context
        filtered_courses = courses.copy()
        
        if context.course_code:
            filtered_courses = [c for c in filtered_courses if 
                               context.course_code.lower() in c.get("course_code", "").lower()]
        
        if context.course_name and not filtered_courses:
            filtered_courses = [c for c in courses if 
                               context.course_name.lower() in c.get("course_name", "").lower() or 
                               context.course_name.lower() in c.get("heading", "").lower()]
        
        # If no specific filters matched, try keywords
        if not filtered_courses and context.keywords:
            for keyword in context.keywords:
                # Search in text, heading, and keywords fields
                keyword_matches = []
                for course in courses:
                    text = course.get("text", "").lower()
                    heading = course.get("heading", "").lower()
                    course_keywords = [k.lower() for k in course.get("keywords", [])]
                    
                    if (keyword.lower() in text or 
                        keyword.lower() in heading or 
                        any(keyword.lower() in k for k in course_keywords)):
                        keyword_matches.append(course)
                
                if keyword_matches:
                    filtered_courses.extend(keyword_matches)
            
            # Remove duplicates if we found multiple matches
            if filtered_courses:
                seen_ids = set()
                filtered_courses = [c for c in filtered_courses if c.get("id") not in seen_ids and not seen_ids.add(c.get("id"))]
        
        # Limit results
        filtered_courses = filtered_courses[:5]
        
        context.search_results = filtered_courses
        logger.info(f"Found {len(filtered_courses)} course entries matching the query")
        
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving course information: {e}")
        return context

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
        # Create initial context with available fields
        context = CourseQueryContext(
            user_query=query,
            response_language=language
        )
        
        # Process the query to extract entities for the context
        # This would require separate entity extraction logic
        # For now, we'll use the agent directly with the query
        
        # Add search results to context
        context = await add_course_search_context(context)
        
        # Get the course agent with the dynamically selected model
        course_agent = get_course_agent()
        
        # Generate response using the agent
        result = await course_agent.run(context)
        
        return result.output.answer
        
    except Exception as e:
        logger.error(f"Error generating course response: {e}")
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