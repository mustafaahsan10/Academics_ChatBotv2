import os
import json
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any
import pydantic_ai
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import pandas as pd
from pathlib import Path
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "study_resources"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

class ResourceQueryContext(BaseModel):
    """Context for a study resource query"""
    user_query: str = Field(..., description="The user's original query about study resources")
    course_code: Optional[str] = Field(None, description="Course code mentioned in the query (e.g., 'CS101')")
    resource_type: Optional[str] = Field(None, description="Type of resource mentioned (e.g., 'textbook', 'slides', 'notes')")
    topic: Optional[str] = Field(None, description="Specific topic mentioned in the query")
    author_mentioned: Optional[str] = Field(None, description="Author or publisher mentioned in the query")
    format_mentioned: Optional[str] = Field(None, description="Format mentioned (e.g., 'online', 'pdf', 'physical')")
    availability: Optional[str] = Field(None, description="Availability mentioned (e.g., 'library', 'online access')")
    is_recommendation_request: bool = Field(False, description="Whether the query is asking for recommendations")
    keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
    search_results: Optional[List[Dict[Any, Any]]] = Field(None, description="Relevant resource information retrieved")
    response_language: str = Field("English", description="Language to respond in (English or Arabic)")

class ResourceResponse(BaseModel):
    """Structured response for study resource queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query about study resources")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions related to resources")

def get_resource_agent():
    """
    Get the AI agent for study resources with the model selected by the user
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
    
    logger.info(f"Using model: {model} for study resources")
    
    return pydantic_ai.Agent(
        model,
        input_type=ResourceQueryContext,
        output_type=ResourceResponse,
        system_prompt="""You are a knowledgeable university assistant specializing in study resources.
Your goal is to provide accurate, helpful information about textbooks, study materials, and learning resources.

When responding to queries about study resources:
1. Prioritize official course materials when available
2. Include specific details about resources (author, edition, format)
3. Mention where resources can be accessed (library, online platforms)
4. Note any supplementary materials that might be helpful
5. If recommending alternatives, explain why they might be beneficial

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your guidance for their learning, so be specific, accurate, and helpful.
"""
    )

async def add_resource_search_context(context: ResourceQueryContext) -> ResourceQueryContext:
    """
    Add relevant search results to the context for study resource queries
    
    Args:
        context: The current query context
        
    Returns:
        Updated context with search results
    """
    try:
        # Load study resources data
        data_path = Path("data/processed/study_resources.json")
        if not data_path.exists():
            logger.warning(f"Study resources data file not found at {data_path}")
            return context
            
        with open(data_path, "r", encoding="utf-8") as f:
            resources = json.load(f)
        
        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(resources)
        
        # Apply filters based on context
        filtered_df = df.copy()
        
        if context.course_code:
            filtered_df = filtered_df[filtered_df["course_code"].str.contains(context.course_code, case=False, na=False)]
        
        if context.resource_type:
            filtered_df = filtered_df[filtered_df["resource_type"].str.contains(context.resource_type, case=False, na=False)]
            
        if context.topic:
            filtered_df = filtered_df[filtered_df["topic"].str.contains(context.topic, case=False, na=False)]
            
        if context.author_mentioned and "author" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["author"].str.contains(context.author_mentioned, case=False, na=False)]
            
        if context.format_mentioned and "format" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["format"].str.contains(context.format_mentioned, case=False, na=False)]
            
        if context.availability and "availability" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["availability"].str.contains(context.availability, case=False, na=False)]
        
        # If no specific filters applied, use keywords for general search
        if len(filtered_df) == len(df) and context.keywords:
            # Search across all text columns
            text_columns = ["course_code", "course_name", "resource_type", "title", "topic", "description"]
            if "author" in filtered_df.columns:
                text_columns.append("author")
            
            for keyword in context.keywords:
                mask = pd.Series(False, index=filtered_df.index)
                for col in text_columns:
                    if col in filtered_df.columns:
                        mask = mask | filtered_df[col].astype(str).str.contains(keyword, case=False, na=False)
                filtered_df = filtered_df[mask]
        
        # Limit results
        filtered_df = filtered_df.head(10)
        
        # Convert results to list of dictionaries
        context.search_results = filtered_df.to_dict(orient="records")
        
        logger.info(f"Found {len(context.search_results)} study resources matching the query")
        
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving study resources: {e}")
        return context

async def get_resource_response(query: str, language: str = "English") -> str:
    """
    Get a response for a study resource query
    
    Args:
        query: The user's question about study resources
        language: The language to respond in
        
    Returns:
        Formatted response string
    """
    try:
        # Create initial context with available fields
        context = ResourceQueryContext(
            user_query=query,
            response_language=language
        )
        
        # Process the query to extract entities for the context
        # This would require separate entity extraction logic
        # For now, we'll use the agent directly with the query
        
        # Add search results to context
        context = await add_resource_search_context(context)
        
        # Get the resource agent with the dynamically selected model
        resource_agent = get_resource_agent()
        
        # Generate response using the agent
        result = await resource_agent.run(context)
        
        return result.output.answer
        
    except Exception as e:
        logger.error(f"Error generating resource response: {e}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة استفسارك عن المصادر الدراسية. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your study resources query. Please try again."

def get_resource_response_sync(query: str, language: str = "English") -> str:
    """
    Synchronous wrapper for get_resource_response
    
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
        
    return loop.run_until_complete(get_resource_response(query, language))