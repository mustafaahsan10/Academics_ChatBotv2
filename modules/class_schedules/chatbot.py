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
COLLECTION_NAME = "class_schedules"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Simplified context model for class schedules queries
class ScheduleQueryContext(BaseModel):
    """Context for a class schedule query"""
    user_query: str = Field(..., description="The user's original query about class schedules")
    course_code: Optional[str] = Field(None, description="Course code mentioned in the query (e.g., 'CSC 226')")
    course_name: Optional[str] = Field(None, description="Course name mentioned (e.g., 'Database Systems')")
    day: Optional[str] = Field(None, description="Day of the week mentioned (e.g., 'Monday', 'Tuesday')")
    time: Optional[str] = Field(None, description="Time mentioned in the query")
    keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
    search_results: Optional[List[Dict[Any, Any]]] = Field(None, description="Relevant class schedule information retrieved")
    response_language: str = Field("English", description="Language to respond in (English or Arabic)")

# Response model for class schedules
class ScheduleResponse(BaseModel):
    """Structured response for class schedule queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query about class schedules")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions related to schedules")

# Function to get an AI agent with the dynamically selected model
def get_schedule_agent():
    """
    Get the AI agent for class schedules with the model selected by the user
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
    
    logger.info(f"Using model: {model} for class schedules")
    
    return pydantic_ai.Agent(
        model,
        input_type=ScheduleQueryContext,
        output_type=ScheduleResponse,
        system_prompt="""You are a knowledgeable university assistant specializing in class schedules.
Your goal is to provide accurate, helpful information about class timings, locations, and related details.

When responding to queries about class schedules:
1. Be precise about days and times
2. When referencing specific courses, include the course code and name
3. Include all relevant information found in the search results
4. If information is missing or uncertain, clearly state this

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your accuracy for their attendance, so be as specific and clear as possible.
"""
    )

async def add_schedule_search_context(context: ScheduleQueryContext) -> ScheduleQueryContext:
    """
    Add relevant search results to the context for class schedule queries
    
    Args:
        context: The current query context
        
    Returns:
        Updated context with search results
    """
    try:
        # Load class schedule data
        data_path = Path("data/processed/class_schedules.json")
        if not data_path.exists():
            logger.warning(f"Class schedule data file not found at {data_path}")
            return context
            
        with open(data_path, "r", encoding="utf-8") as f:
            schedules = json.load(f)
        
        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(schedules)
        
        # Apply filters based on context
        filtered_df = df.copy()
        
        if context.course_code:
            filtered_df = filtered_df[filtered_df["course_code"].str.contains(context.course_code, case=False, na=False)]
        
        if context.course_name:
            filtered_df = filtered_df[filtered_df["course_name"].str.contains(context.course_name, case=False, na=False)]
        
        if context.day:
            # Need to search within the sessions array for the day
            filtered_df = filtered_df[filtered_df.apply(
                lambda row: any(context.day.lower() in session.get("day", "").lower() 
                              for session in row.get("sessions", [])), 
                axis=1
            )]
            
        if context.time:
            # Need to search within the sessions array for the time
            filtered_df = filtered_df[filtered_df.apply(
                lambda row: any(context.time in session.get("time_slot", "") 
                              for session in row.get("sessions", [])), 
                axis=1
            )]
            
        # If no specific filters applied, use keywords for general search
        if len(filtered_df) == len(df) and context.keywords:
            # Search in text field and keywords field
            keyword_mask = pd.Series(False, index=filtered_df.index)
            
            for keyword in context.keywords:
                # Search in text field
                text_mask = filtered_df["text"].str.contains(keyword, case=False, na=False)
                
                # Search in keywords list field
                keywords_mask = filtered_df.apply(
                    lambda row: any(keyword.lower() in kw.lower() for kw in row.get("keywords", [])),
                    axis=1
                )
                
                # Search in heading
                heading_mask = filtered_df["heading"].str.contains(keyword, case=False, na=False)
                
                # Combine masks
                keyword_mask = keyword_mask | text_mask | keywords_mask | heading_mask
            
            filtered_df = filtered_df[keyword_mask]
        
        # Limit results
        filtered_df = filtered_df.head(5)
        
        # Convert results to list of dictionaries
        context.search_results = filtered_df.to_dict(orient="records")
        
        logger.info(f"Found {len(context.search_results)} schedule entries matching the query")
        
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving schedule information: {e}")
        return context

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
        # Create initial context with available fields
        context = ScheduleQueryContext(
            user_query=query,
            response_language=language
        )
        
        # Process the query to extract entities for the context
        # This would require separate entity extraction logic
        # For now, we'll use the agent directly with the query
        
        # Add search results to context
        context = await add_schedule_search_context(context)
        
        # Get the schedule agent with the dynamically selected model
        schedule_agent = get_schedule_agent()
        
        # Generate response using the agent
        result = await schedule_agent.run(context)
        
        return result.output.answer
        
    except Exception as e:
        logger.error(f"Error generating schedule response: {e}")
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