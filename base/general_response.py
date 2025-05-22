import os
import logging
from dotenv import load_dotenv
from typing import List
import pydantic_ai
from pydantic import BaseModel, Field
import streamlit as st
import json
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GeneralResponse(BaseModel):
    """Response for general queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query")

def get_general_response(query: str, language: str = "English") -> str:
    """
    Get a general response for user queries without vector search
    
    Args:
        query: The user's question
        language: The language to respond in
        
    Returns:
        Formatted response string
    """
    try:
        logger.info(f"Processing general query: {query}")
        
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
        
        # Prepare system message for general responses
        system_message = f"""You are a helpful and friendly university assistant chatbot.
Your goal is to provide accurate, helpful information to university students.

{language_instruction}
Be conversational, friendly, and direct in your responses.

Remember that you're interacting with university students, so focus on being helpful and informative.
If asked about specific university information that would require specialized knowledge, 
provide general guidance based on common knowledge about universities, academia, and student life.
"""
        
        # Generate response based on model type
        if use_openrouter:
            # Use OpenRouter API
            import requests
            
            openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY")
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
                api_key=st.secrets["OPENAI_API_KEY"],
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