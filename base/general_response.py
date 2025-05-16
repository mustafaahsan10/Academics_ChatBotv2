import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import pydantic_ai
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
from qdrant_client import QdrantClient
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize embeddings and Qdrant client
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Available collections to search
AVAILABLE_COLLECTIONS = [
    "course_information", 
    "class_schedules", 
    "exam_alerts", 
    "study_resources", 
    "professors"
]

class GeneralSearchResult(BaseModel):
    """Result from general search across all collections"""
    context: str = Field(..., description="Context information retrieved from search")
    sources: List[str] = Field(default_factory=list, description="Sources of the information")

class GeneralResponse(BaseModel):
    """Response for general queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query")

def search_all_collections(query: str, top_k_per_collection: int = 10) -> Dict[str, List[Any]]:
    """
    Search across all available collections
    
    Args:
        query: The search query
        top_k_per_collection: Number of results to return per collection
        
    Returns:
        Dictionary mapping collection names to search results
    """
    try:
        logger.info(f"Searching across all collections for: {query}")
        
        # Get query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Search all collections
        all_results = {}
        
        for collection_name in AVAILABLE_COLLECTIONS:
            try:
                # Check if collection exists
                collections = qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if collection_name not in collection_names:
                    logger.info(f"Collection {collection_name} does not exist, skipping")
                    continue
                
                # Perform vector search
                results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=top_k_per_collection
                )
                
                if results:
                    all_results[collection_name] = results
                    logger.info(f"Found {len(results)} results in {collection_name}")
                
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
                continue
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error searching across collections: {e}")
        return {}

def format_search_results(all_results: Dict[str, List[Any]]) -> GeneralSearchResult:
    """
    Format search results from all collections into a single context
    
    Args:
        all_results: Dictionary mapping collection names to search results
        
    Returns:
        Formatted context and sources
    """
    if not all_results:
        return GeneralSearchResult(
            context="No information found related to your query.",
            sources=[]
        )
    
    context_parts = []
    sources = []
    
    for collection_name, results in all_results.items():
        if not results:
            continue
        
        # Add collection header
        context_parts.append(f"\n## Information from {collection_name.replace('_', ' ').title()}\n")
        sources.append(collection_name)
        
        for i, hit in enumerate(results):
            payload = hit.payload
            
            # Format based on collection type
            if collection_name == "exam_alerts":
                # Format exam info
                item_text = f"Course: {payload.get('course_code', 'N/A')} - {payload.get('course_name', 'Unknown')}\n"
                item_text += f"Type: {payload.get('type', 'Final Exam')}\n"
                item_text += f"Date: {payload.get('date', 'Not specified')}\n"
                item_text += f"Time: {payload.get('time', 'Not specified')}\n"
                
            elif collection_name == "class_schedules":
                # Format schedule info
                item_text = f"Course: {payload.get('course_code', 'N/A')} - {payload.get('course_name', 'Unknown')}\n"
                sessions = payload.get("sessions", [])
                if sessions:
                    item_text += "Schedule:\n"
                    for session in sessions:
                        day = session.get("day", "Unknown")
                        time = session.get("time_slot", "Unknown time")
                        item_text += f"- {day}: {time}\n"
                        
            elif collection_name == "professors":
                # Format professor info
                item_text = f"Name: {payload.get('name', 'Unknown')}\n"
                item_text += f"Department: {payload.get('department', 'Not specified')}\n"
                item_text += f"Office Hours: {payload.get('office_hours', 'Not specified')}\n"
                
            elif collection_name == "course_information":
                # Format course info
                metadata = payload.get("metadata", {})
                item_text = ""
                
                if "heading" in metadata:
                    item_text += f"Section: {metadata['heading']}\n"
                
                course_details = metadata.get("course_details", {})
                if course_details:
                    if course_details.get("code"):
                        item_text += f"Course Code: {course_details['code']}\n"
                    if course_details.get("name"):
                        item_text += f"Course Name: {course_details['name']}\n"
                
                if payload.get("text"):
                    item_text += f"Details: {payload.get('text')}\n"
                
            elif collection_name == "study_resources":
                # Format study resource info
                item_text = f"Resource: {payload.get('title', 'Unknown resource')}\n"
                item_text += f"Type: {payload.get('resource_type', 'Unknown type')}\n"
                if payload.get("text"):
                    item_text += f"Details: {payload.get('text')}\n"
                
            else:
                # Generic format for unknown collection types
                if payload.get("text"):
                    item_text = payload.get("text")
                else:
                    item_text = str(payload)
            
            # Add score
            item_text += f"(Relevance: {hit.score:.2f})\n"
            
            # Add to context
            context_parts.append(f"{i+1}. {item_text}")
    
    context = "\n".join(context_parts)
    return GeneralSearchResult(
        context=context,
        sources=sources
    )

def get_general_response(query: str, language: str = "English") -> str:
    """
    Get a general response by searching across all collections
    
    Args:
        query: The user's question
        language: The language to respond in
        
    Returns:
        Formatted response string
    """
    try:
        logger.info(f"Processing general query: {query}")
        
        # Search across all collections with explicitly higher result count
        all_results = search_all_collections(query, top_k_per_collection=15)
        
        # Format search results
        search_result = format_search_results(all_results)
        context = search_result.context
        
        # Prepare language instruction
        language_instruction = ""
        if language.lower() == "arabic":
            language_instruction = "Respond in fluent Arabic."
        else:
            language_instruction = "Respond in clear English."
        
        # Get model from Streamlit session state if available
        model_id = "gpt-4o-mini"  # Default model
        use_openrouter = False
        
        if "model" in st.session_state:
            model_id = st.session_state.model
            if "use_openrouter" in st.session_state:
                use_openrouter = st.session_state.use_openrouter
        
        logger.info(f"Using model: {model_id}, OpenRouter: {use_openrouter}")
        
        # Prepare system message
        system_message = f"""You are a knowledgeable university assistant.
Your goal is to provide accurate, helpful information to university students based on the following search results:

{context}

Provide a comprehensive response addressing the user's query. {language_instruction}
Be conversational but direct, and don't mention that your response is based on search results.

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about universities, academia, and student life. Never say phrases like "I don't have information on this" or "this isn't in my database".

Always maintain a helpful tone and try to address the user's query as best as possible without revealing any limitations in your knowledge base.
"""
        
        # Generate response based on model type
        if use_openrouter:
            # Use OpenRouter API
            import requests
            
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key not found")
            
            # Format model name for OpenRouter
            if model_id.startswith("claude-3"):
                openrouter_model = f"anthropic/{model_id}"
            elif model_id.startswith("gemini"):
                openrouter_model = f"google/{model_id}"
            else:
                openrouter_model = model_id
            
            logger.info(f"Using OpenRouter model: {openrouter_model}")
            
            # Make API request
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://university-chatbot.com",
                "X-Title": "University Chatbot"
            }
            
            payload = {
                "model": openrouter_model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ]
            }
            
            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(payload))
                
                # Check if request was successful
                if response.status_code == 200:
                    response_data = response.json()
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        return response_data["choices"][0]["message"]["content"]
                    else:
                        raise ValueError("Invalid response format from OpenRouter")
                else:
                    logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                    raise ValueError(f"OpenRouter API error: {response.status_code}")
            except Exception as e:
                logger.error(f"Error with OpenRouter API: {e}")
                logger.info("Falling back to standard OpenAI model")
                # Fall back to standard OpenAI model
                use_openrouter = False
                model_id = "gpt-4o-mini"  # Use a reliable fallback model
        
        # Use pydantic_ai Agent for OpenAI models
        if not use_openrouter:
            general_agent = pydantic_ai.Agent(
                model=model_id,
                api_key=os.getenv("OPENAI_API_KEY"),
                system_prompt=system_message,
                output_type=GeneralResponse
            )
            
            user_message = query
            if language.lower() == "arabic":
                user_message = f"{query} (Please respond in Arabic)"
                
            response = general_agent.run_sync(user_message)
            
            if hasattr(response, 'output'):
                if hasattr(response.output, 'answer'):
                    return response.output.answer
                else:
                    return str(response.output)
            else:
                return str(response)
            
    except Exception as e:
        import traceback
        logger.error(f"Error generating general response: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة استفسارك. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your query. Please try again."

def get_general_response_sync(query: str, language: str = "English") -> str:
    """
    Synchronous wrapper for get_general_response
    
    Args:
        query: The user's question
        language: The language to respond in
        
    Returns:
        Formatted response string
    """
    return get_general_response(query, language) 