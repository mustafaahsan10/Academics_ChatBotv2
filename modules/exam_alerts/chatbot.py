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
from datetime import datetime
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "exam_alerts"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

class ExamQueryContext(BaseModel):
    """Context for an exam/deadline query"""
    user_query: str = Field(..., description="The user's original query about exams or deadlines")
    course_code: Optional[str] = Field(None, description="Course code mentioned in the query (e.g., 'CS101')")
    exam_type: Optional[str] = Field(None, description="Type of exam mentioned (e.g., 'midterm', 'final', 'quiz')")
    date_mentioned: Optional[str] = Field(None, description="Date or time period mentioned (e.g., 'next week', 'December')")
    location_mentioned: Optional[str] = Field(None, description="Location mentioned for the exam")
    assessment_type: Optional[str] = Field(None, description="Type of assessment (e.g., 'assignment', 'project', 'presentation')")
    professor_name: Optional[str] = Field(None, description="Professor name mentioned in the query")
    is_upcoming_request: bool = Field(False, description="Whether the query is about upcoming exams or deadlines")
    keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
    search_results: Optional[List[Dict[Any, Any]]] = Field(None, description="Relevant exam information retrieved")
    response_language: str = Field("English", description="Language to respond in (English or Arabic)")

class ExamResponse(BaseModel):
    """Structured response for exam/deadline queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query about exams or deadlines")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions related to exams")

def get_exam_agent():
    """
    Get the AI agent for exam alerts with the model selected by the user
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
    
    logger.info(f"Using model: {model} for exam alerts")
    
    return pydantic_ai.Agent(
        model,
        input_type=ExamQueryContext,
        output_type=ExamResponse,
        system_prompt="""You are a knowledgeable university assistant specializing in exam schedules and academic deadlines.
Your goal is to provide accurate, timely information about exams, assignments, and important academic dates.

When responding to queries about exams and deadlines:
1. Prioritize upcoming events that are most relevant to the student
2. Include specific dates, times, and locations when available
3. Mention any special requirements (e.g., materials allowed, submission format)
4. Note the weight or importance of assessments when relevant
5. If any information is missing or uncertain, clearly state this

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your accuracy for their academic planning, so be specific and clear.
"""
    )

async def add_exam_search_context(context: ExamQueryContext) -> ExamQueryContext:
    """
    Add relevant search results to the context for exam and deadline queries
    
    Args:
        context: The current query context
        
    Returns:
        Updated context with search results
    """
    try:
        # Load exam and deadline data
        data_path = Path("data/processed/exams_deadlines.json")
        if not data_path.exists():
            logger.warning(f"Exam data file not found at {data_path}")
            return context
            
        with open(data_path, "r", encoding="utf-8") as f:
            exams = json.load(f)
        
        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(exams)
        
        # Apply filters based on context
        filtered_df = df.copy()
        
        if context.course_code:
            filtered_df = filtered_df[filtered_df["course_code"].str.contains(context.course_code, case=False, na=False)]
        
        if context.exam_type:
            filtered_df = filtered_df[filtered_df["type"].str.contains(context.exam_type, case=False, na=False)]
            
        if context.professor_name:
            filtered_df = filtered_df[filtered_df["instructor"].str.contains(context.professor_name, case=False, na=False)]
            
        if context.location_mentioned:
            filtered_df = filtered_df[filtered_df["location"].str.contains(context.location_mentioned, case=False, na=False)]
            
        if context.assessment_type:
            filtered_df = filtered_df[filtered_df["type"].str.contains(context.assessment_type, case=False, na=False)]
        
        # Filter for upcoming exams if requested
        if context.is_upcoming_request and "date" in filtered_df.columns:
            # Convert dates to datetime objects for comparison
            today = datetime.now()
            if "date" in filtered_df.columns:
                filtered_df = filtered_df[pd.to_datetime(filtered_df["date"], errors="coerce") >= today]
        
        # If no specific filters applied, use keywords for general search
        if len(filtered_df) == len(df) and context.keywords:
            # Search across all text columns
            text_columns = ["course_code", "course_name", "type", "instructor", "location", "notes"]
            
            for keyword in context.keywords:
                mask = pd.Series(False, index=filtered_df.index)
                for col in text_columns:
                    if col in filtered_df.columns:
                        mask = mask | filtered_df[col].astype(str).str.contains(keyword, case=False, na=False)
                filtered_df = filtered_df[mask]
        
        # Sort by date if available
        if "date" in filtered_df.columns:
            filtered_df["date"] = pd.to_datetime(filtered_df["date"], errors="coerce")
            filtered_df = filtered_df.sort_values(by="date")
        
        # Limit results
        filtered_df = filtered_df.head(10)
        
        # Convert results to list of dictionaries
        context.search_results = filtered_df.to_dict(orient="records")
        
        logger.info(f"Found {len(context.search_results)} exam/deadline entries matching the query")
        
        return context
        
    except Exception as e:
        logger.error(f"Error retrieving exam information: {e}")
        return context

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
        # Create initial context with available fields
        context = ExamQueryContext(
            user_query=query,
            response_language=language
        )
        
        # Process the query to extract entities for the context
        # This would require separate entity extraction logic
        # For now, we'll use the agent directly with the query
        
        # Add search results to context
        context = await add_exam_search_context(context)
        
        # Get the exam agent with the dynamically selected model
        exam_agent = get_exam_agent()
        
        # Generate response using the agent
        result = await exam_agent.run(context)
        
        return result.output.answer
        
    except Exception as e:
        logger.error(f"Error generating exam response: {e}")
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