import os
import json
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
import pydantic_ai
from pydantic import BaseModel, Field
from pathlib import Path
import streamlit as st
import pandas as pd
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "study_resources"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Search request and response models
class ResourceSearchRequest(BaseModel):
    """Request parameters for searching study resources"""
    query: str = Field(..., description="The query to search for study resources")
    language: str = Field("English", description="The language to respond in")

class ResourceSearchResult(BaseModel):
    """Result from searching study resources"""
    context: str = Field(..., description="Context information retrieved from search")

# Query context model
# class ResourceQueryContext(BaseModel):
#     """Context for a study resource query"""
#     user_query: str = Field(..., description="The user's original query about study resources")
#     course_name: Optional[str] = Field(None, description="Course name mentioned in the query (e.g., 'Introduction to Computer Science')")
#     resource_type: Optional[str] = Field(None, description="Type of resource mentioned (e.g., 'study material', 'exam links')")
#     topic: Optional[str] = Field(None, description="Specific topic mentioned in the query")
#     keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
#     search_results: Optional[List[Dict[Any, Any]]] = Field(None, description="Relevant resource information retrieved")
#     response_language: str = Field("English", description="Language to respond in (English or Arabic)")

# Response model
class ResourceResponse(BaseModel):
    """Structured response for study resource queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query about study resources")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions related to resources")

def get_resource_agent():
    """
    Get the AI agent for study resources with the model selected by the user
    """
    # Get model from Streamlit session state if available, otherwise use default
    model_id = "gpt-4o-mini"  # Default model
    use_openrouter = False
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if "model" in st.session_state:
        model_id = st.session_state.model
        # Check if we should use OpenRouter based on session state
        if "use_openrouter" in st.session_state and st.session_state.use_openrouter:
            use_openrouter = True
    
    logger.info(f"Using model: {model_id} for study resources with tools")
    if use_openrouter:
        logger.info(f"Using OpenRouter API for model: {model_id}")
    
    # Create the agent with tools
    system_prompt = """You are a knowledgeable university assistant specializing in study resources.
Your goal is to provide accurate, helpful information about study materials and learning resources for various courses.

You have access to a search_resources tool that can find information about university course study materials and exam links.
Always use this tool to look up information before answering questions.

When responding to queries about study resources:
1. Prioritize official course materials when available
2. Include specific details about the study materials available
3. Mention exam links when relevant
4. Suggest appropriate study methods for specific courses
5. If recommending alternatives, explain why they might be beneficial

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about university study materials and resources. Never say phrases like "I don't have information on this" or "this isn't in my database". Always maintain a helpful tone and try to address the user's query as best as possible.

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your guidance for their learning, so be specific, accurate, and helpful.
"""

    # Create the agent with appropriate configuration
    resource_agent = pydantic_ai.Agent(
        model=model_id,
        tools=[search_resources],
        system_prompt=system_prompt,
        openai_api_key=openrouter_api_key if use_openrouter else openai_api_key,
        openai_api_base="https://openrouter.ai/api/v1" if use_openrouter else None
    )
    return resource_agent

def search_resources(request: ResourceSearchRequest) -> ResourceSearchResult:
    """
    Search for study resources based on the query
    
    Args:
        request: The search request with query and language
        
    Returns:
        Results matching the search query
    """
    try:
        # Process the query to extract information
        entities = extract_resource_entities(request.query)
        
        # Load and search the resource data
        resources_data = load_and_search_resources(request.query, entities)
        
        # Format the results
        context = format_resource_results(resources_data)
        
        return ResourceSearchResult(
            context=context
        )
            
    except Exception as e:
        logger.error(f"Error searching study resources: {e}")
        return ResourceSearchResult(
            context="Error searching for study resources."
        )

def extract_resource_entities(query: str) -> Dict[str, Any]:
    """Extract study resource related entities from the query"""
    entities = {
        "course_name": None,
        "resource_type": None,
        "keywords": []
    }
    
    # List of courses to look for
    courses = [
        "Introduction to Computer Science",
        "Data Structures and Algorithms",
        "Database Systems",
        "Operating Systems",
        "Computer Networks",
        "Software Engineering",
        "Web Development",
        "Artificial Intelligence",
        "Machine Learning",
        "Cybersecurity"
    ]
    
    # Check for course names in the query
    lower_query = query.lower()
    for course in courses:
        if course.lower() in lower_query:
            entities["course_name"] = course
            entities["keywords"].append(course.lower())
            break
    
    # Check for resource type
    if any(term in lower_query for term in ["exam", "test", "quiz"]):
        entities["resource_type"] = "exam links"
        entities["keywords"].append("exam")
    elif any(term in lower_query for term in ["study", "material", "textbook", "note"]):
        entities["resource_type"] = "study material"
        entities["keywords"].append("study material")
    
    # Extract important keywords
    important_terms = ["textbook", "study", "material", "resource", "exam", "link", "learn", 
                       "course", "note", "recommend", "help", "understand", "practice"]
    
    for term in important_terms:
        if term in lower_query and term not in entities["keywords"]:
            entities["keywords"].append(term)
    
    logger.info(f"Extracted entities: {entities}")
    return entities

def load_and_search_resources(query: str, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load study resources data and search based on the query and entities
    
    Args:
        query: The original query
        entities: Extracted entities from the query
        
    Returns:
        List of matching resources
    """
    try:
        # Load study resources data - updated filename
        data_path = Path("data/processed/study_resource.json")
        if not data_path.exists():
            logger.warning(f"Study resources data file not found at {data_path}")
            return []
            
        with open(data_path, "r", encoding="utf-8") as f:
            resources = json.load(f)
        
        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(resources)
        
        # Rename columns to match our code's expectations
        if "Course Name" in df.columns:
            df.rename(columns={
                "Course Name": "course_name",
                "Study Material": "description",
                "Exam Links": "exam_links"
            }, inplace=True)
        
        # Apply filters based on entities
        filtered_df = df.copy()
        
        # Filter by course name
        if entities["course_name"]:
            filtered_df = filtered_df[filtered_df["course_name"].str.contains(entities["course_name"], case=False, na=False)]
        
        # Filter by resource type
        if entities["resource_type"]:
            if "study material" in entities["resource_type"].lower():
                # Ensure description has content
                filtered_df = filtered_df[filtered_df["description"].notna()]
            elif "exam" in entities["resource_type"].lower():
                # Ensure exam_links has content
                filtered_df = filtered_df[filtered_df["exam_links"].notna()]
        
        # If no specific filters applied, use keywords for general search
        if len(filtered_df) == len(df) and entities["keywords"]:
            # Search across all text columns
            text_columns = ["course_name", "description", "exam_links"]
            
            for keyword in entities["keywords"]:
                mask = pd.Series(False, index=filtered_df.index)
                for col in text_columns:
                    if col in filtered_df.columns:
                        mask = mask | filtered_df[col].astype(str).str.contains(keyword, case=False, na=False)
                filtered_df = filtered_df[mask]
        
        # Limit results and convert to list of dictionaries
        return filtered_df.head(10).to_dict(orient="records")
        
    except Exception as e:
        logger.error(f"Error loading and searching resources: {e}")
        return []

def format_resource_results(results: List[Dict[str, Any]]) -> str:
    """Format resource search results into a context string"""
    if not results:
        return "No specific study resources found related to your query."
    
    context_parts = []
    
    for i, resource in enumerate(results):
        # Create a formatted result for each resource
        resource_parts = []
        
        # Add course name
        if "course_name" in resource:
            resource_parts.append(f"Course: {resource['course_name']}")
        
        # Add study materials
        if "description" in resource and resource["description"]:
            resource_parts.append(f"Study Materials: {resource['description']}")
        
        # Add exam links
        if "exam_links" in resource and resource["exam_links"]:
            resource_parts.append(f"Exam Resources: {resource['exam_links']}")
        
        # Combine parts and add to context
        context_parts.append("\n".join(resource_parts))
    
    return "\n\n---\n\n".join(context_parts)

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
        # Get model information from session state
        model_id = "gpt-4o-mini"  # Default model
        use_openrouter = False
        
        if "model" in st.session_state:
            model_id = st.session_state.model
            if "use_openrouter" in st.session_state:
                use_openrouter = st.session_state.use_openrouter
                
        logger.info(f"Processing query with model: {model_id}, using OpenRouter: {use_openrouter}")
        
        # Prepare user message with language preference
        user_message = query
        if language.lower() == "arabic":
            user_message = f"{query} (Please respond in Arabic)"
        else:
            user_message = f"{query} (Please respond in English)"
        
        if use_openrouter:
            # Direct OpenRouter API approach for better control
            # Extract entities and search for resources
            entities = extract_resource_entities(query)
            search_results = load_and_search_resources(query, entities)
            context = format_resource_results(search_results)
            
            # Prepare system message with context
            system_message = """You are a knowledgeable university assistant specializing in study resources.
Your goal is to provide accurate, helpful information about study materials and learning resources for various courses
based on the following information:

""" + context + """

When responding to queries about study resources:
1. Prioritize official course materials when available
2. Include specific details about the study materials available
3. Mention exam links when relevant
4. Suggest appropriate study methods for specific courses
5. If recommending alternatives, explain why they might be beneficial

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
                "HTTP-Referer": "https://university-chatbot.com",  # Update with your site URL
                "X-Title": "University Chatbot"  # Update with your site name
            }
            
            # Format the model name for OpenRouter
            # Claude models need "anthropic/" prefix, and Gemini needs "google/" prefix
            if model_id in ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"]:
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
        resource_agent = get_resource_agent()
        response = await resource_agent.run(user_message)
        
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
        logger.error(f"Error generating resource response: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
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