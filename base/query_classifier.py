import logging
from dotenv import load_dotenv
from typing import List, Optional
import pydantic_ai
from pydantic import BaseModel, Field
import os
import streamlit as st
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except:
    pass
class QueryClassification(BaseModel):
    """Classification of a user query to determine which module should handle it"""
    user_query: str = Field(..., description="The original user query")
    module: str = Field(..., description="The module that should handle this query", 
                       examples=["course_information", "class_schedules", "exam_alerts", "study_resources", "professors", "general_response"])
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level of this classification (0.0 to 1.0)")
    reasoning: str = Field(..., description="Explanation of why this module was selected")
    extracted_entities: Optional[dict] = Field(None, description="Key entities extracted from the query")
    keywords: List[str] = Field(default_factory=list, description="Important keywords from the query")
classifier_agent = None
def get_classifier_agent():
    """Get or create the classifier agent with proper API key setup."""
    global classifier_agent
    
    if classifier_agent is None:
        # Set API key from Streamlit secrets
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        
        classifier_agent = pydantic_ai.Agent(
    "openai:gpt-4.1-nano",  # Can be configured based on environment
    output_type=QueryClassification,
    system_prompt="""
    You are a specialized query classifier for a university academic chatbot system.
    Your task is to analyze user queries and route them to the most appropriate module.

    Available modules:
    1. course_information - For queries about course content, prerequisites, credit hours, descriptions, faculty structure, departments, majors, degree programs, university academic structure
    2. class_schedules - For queries about when and where classes meet, timetables, room numbers, semester start/end dates, lecture times, final exam schedules
    3. exam_alerts - For queries about exam dates, deadlines, assignment due dates, assessments
    4. study_resources - For queries about textbooks, study materials, online resources
    5. professors - For queries about specific faculty members, office hours, contact details, research interests
    6. library - For queries about library resources, books, availability, borrowing, returning, fees, etc.

    Analyze each query carefully to determine which module is most appropriate. Some queries may have multiple aspects,
    but choose the primary intent. If a query is ambiguous, choose the most likely module and indicate a lower confidence score.

    For example:
    
    ## Course Information
    - "What are the prerequisites for CS350?" → course_information (high confidence)
    - "Can you tell me about the Computer Science department?" → course_information (high confidence)
    - "What majors does the Faculty of Engineering offer?" → course_information (high confidence)
    - "What are the available faculties at the university?" → course_information (high confidence)
    - "How many credits does a Computer Science major have?" → course_information (high confidence)
    - "What undergraduate programs are available?" → course_information (high confidence)
    - What is the prerequisite to take course csc 226? (high confidence)

    ## Class Schedules
    - "Can I see the class CSC 226 schedule for the upcoming semester?" → class_schedules (high confidence)
    - "What are the lecture times for Database Systems?" → class_schedules (high confidence)

    ## Professors
    - "How can I find Professor Hoda Maalouf's contact information?" → professors (high confidence)
    - "What are Professor Smith's office hours?" → professors (high confidence)
    - "Where is Professor Johnson's office located?" → professors (high confidence)
    - "What courses is Professor Davis currently teaching?" → professors (high confidence)
    - "Who is teaching [Course Name] this semester?" → professors (high confidence)

    ## Exam Alerts
    - "When will the final exam schedule be released?" → exam_alerts (high confidence)
    - "What time is my Database Systems exam?" → exam_alerts (high confidence)
    - "Where will my exam for CSC 226 be held?" → exam_alerts (high confidence)
    
    ## Library
    - Is book 1948 available for renting? (high confidence)
    - What is the cost for renting book “1984”? (high confidence)
    - How much does the book “who moved my cheese” cost? (high confidence)
    - How can I check if a book is available in the library? (high confidence)

    ## General
    - "Hello, how are you today?" → general_response (high confidence)
    - "Thank you for your help!" → general_response (high confidence)
    - "What can this chatbot do?" → general_response (high confidence)

    Always include your reasoning for why you selected a particular module.
"""
)
    
    return classifier_agent
# Initialize the classifier agent


def classify_query_sync(query: str) -> QueryClassification:
    """
    Classify a user query to determine which module should handle it
    
    Args:
        query: The user's query text
        
    Returns:
        A classification object with the module, confidence, and reasoning
    """
    try:
        # Run the classifier to get structured output with synchronous API
        classifier_agent = get_classifier_agent()
        result = classifier_agent.run_sync(query)
        classification = result.output
        
        # Ensure module is one of the allowed values
        allowed_modules = ["course_information", "class_schedules", "exam_alerts", "study_resources", "professors", "library", "general_response"]
        # if classification.module not in allowed_modules:
        #     # Default to course_information if invalid module
        #     classification.module = "general_response"
        #     classification.confidence = min(classification.confidence, 0.4)
        #     classification.reasoning += " (Forced correction: original module classification was invalid)"
        
        logger.info(f"Classified query '{query}' as module '{classification.module}' with confidence {classification.confidence}")
        
        return classification
        
    except Exception as e:
        logger.error(f"Error classifying query: {e}")
        # Default classification in case of error
        return QueryClassification(
            user_query=query,
            module="course_information",  # Default to most general module
            confidence=0.1,
            reasoning=f"Error during classification: {str(e)}",
            keywords=[]
        )