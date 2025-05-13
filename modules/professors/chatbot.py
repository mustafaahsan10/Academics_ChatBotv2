import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
import pydantic_ai
from pydantic import BaseModel, Field
from pathlib import Path
import json
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from utils.common import extract_keywords_from_query, extract_entities_from_query, format_response
from utils.qdrant_helper import qdrant_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Collection name for this module
COLLECTION_NAME = "professors"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Define dependency class for the Pydantic AI agent
class ProfessorQueryContext(BaseModel):
    """Simplified context for professor queries"""
    user_query: str = Field(..., description="The user's original query about professors")
    professor_name: Optional[str] = Field(None, description="Professor name mentioned in the query")
    course_code: Optional[str] = Field(None, description="Course code mentioned in the query")
    keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
    search_results: Optional[List[Dict[Any, Any]]] = Field(None, description="Relevant professor information retrieved")
    response_language: str = Field("English", description="Language to respond in (English or Arabic)")

# Define output model for the agent
class ProfessorResponse(BaseModel):
    """Response to a professor information query"""
    answer: str = Field(..., description="Detailed answer to the user's question about professors")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions related to professors")

# Function to get an AI agent with the dynamically selected model
def get_professor_agent():
    """
    Get the AI agent for professor information with the model selected by the user
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
    
    logger.info(f"Using model: {model} for professor information")
    
    return pydantic_ai.Agent(
        model,
        input_type=ProfessorQueryContext,
        output_type=ProfessorResponse,
        system_prompt="""You are a specialized university faculty information assistant.
Your goal is to provide accurate information about professors, their courses, and contact details.

When responding:
1. Be specific about professor names and titles
2. Include course codes when mentioning which courses they teach
3. Provide clear and accurate information based on the search results
4. If information is missing or uncertain, clearly state this

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your accuracy when seeking information about faculty.
"""
    )

async def add_professor_search_context(context: ProfessorQueryContext) -> ProfessorQueryContext:
    """
    Add relevant search results to the context for professor queries
    
    Args:
        context: The current query context
        
    Returns:
        Updated context with search results
    """
    try:
        # Load professor data
        data_path = Path("data/processed/professors.json")
        if not data_path.exists():
            logger.warning(f"Professors data file not found at {data_path}")
            return context
            
        with open(data_path, "r", encoding="utf-8") as f:
            professors = json.load(f)
        
        # Convert to list if not already
        if isinstance(professors, dict):
            professors = [professors]
        
        # Apply filters based on context
        filtered_professors = professors.copy()
        
        if context.professor_name:
            filtered_professors = [p for p in filtered_professors if 
                                  context.professor_name.lower() in p.get("heading", "").lower() or
                                  context.professor_name.lower() in p.get("text", "").lower()]
        
        if context.course_code:
            filtered_professors = [p for p in filtered_professors if 
                                  context.course_code.lower() in p.get("text", "").lower()]
        
        # If no specific filters matched, try keywords
        if not filtered_professors and context.keywords:
            for keyword in context.keywords:
                # Search in text, heading, and keywords fields
                keyword_matches = []
                for professor in professors:
                    text = professor.get("text", "").lower()
                    heading = professor.get("heading", "").lower()
                    prof_keywords = [k.lower() for k in professor.get("keywords", [])]
                    
                    if (keyword.lower() in text or 
                        keyword.lower() in heading or 
                        any(keyword.lower() in k for k in prof_keywords)):
                        keyword_matches.append(professor)
                
                if keyword_matches:
                    filtered_professors.extend(keyword_matches)
            
            # Remove duplicates if we found multiple matches
            if filtered_professors:
                seen_ids = set()
                filtered_professors = [p for p in filtered_professors if p.get("id") not in seen_ids and not seen_ids.add(p.get("id"))]
        
        # Limit results
        filtered_professors = filtered_professors[:5]
        
        context.search_results = filtered_professors
        logger.info(f"Found {len(filtered_professors)} professor entries matching the query")
        
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving professor information: {e}")
        return context

async def get_professor_response(query: str, language: str = "English") -> str:
    """
    Get a response to a professor information query
    
    Args:
        query: The user's question about professors
        language: The language for the response
        
    Returns:
        Formatted response string
    """
    try:
        # Create initial context with available fields
        context = ProfessorQueryContext(
            user_query=query,
            response_language=language
        )
        
        # Process the query to extract entities for the context
        # This would require separate entity extraction logic
        # For now, we'll use the agent directly with the query
        
        # Add search results to context
        context = await add_professor_search_context(context)
        
        # Get the professor agent with the dynamically selected model
        professor_agent = get_professor_agent()
        
        # Generate response using the agent
        result = await professor_agent.run(context)
        
        return result.output.answer
        
    except Exception as e:
        logger.error(f"Error generating professor response: {e}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة استفسارك عن أعضاء هيئة التدريس. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your faculty information query. Please try again."

def get_professor_response_sync(query: str, language: str = "English") -> str:
    """
    Synchronous wrapper for get_professor_response
    
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
        
    return loop.run_until_complete(get_professor_response(query, language))