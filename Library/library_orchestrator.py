import re
from openai import OpenAI
from typing import Dict, Any, Tuple, Optional
import os
from Library.DB_endpoint import db_endpoint
from modules.classified_chatbot import rag_pipeline_simple
import streamlit as st
# Set up OpenRouter client for intent detection
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
)

def detect_db_query_intent(query: str) -> Tuple[bool, float, str]:
    """
    Detect if the query is a database-specific library query (about books, availability, etc.).
    
    Args:
        query: The user's question
        
    Returns:
        Tuple containing:
        - Boolean indicating if the query should use the database endpoint
        - Confidence score (0-1)
        - Reasoning for the classification
    """
    system_prompt = """
    You are an intent classifier for a university library assistant. Determine if the query is about specific library 
    database information (books, availability, authors, etc.) or a more general library question.
    
    Return ONLY a JSON object with the following structure:
    {
        "is_db_query": true/false,
        "confidence": <float between 0 and 1>,
        "reasoning": "<brief explanation>"
    }
    
    Examples of DATABASE QUERIES (return true):
    - "Do you have any books on machine learning?" → {"is_db_query": true, "confidence": 0.95, "reasoning": "Asking about specific books in the library database"}
    - "Is 'Clean Code' available in the library?" → {"is_db_query": true, "confidence": 0.9, "reasoning": "Asking about availability of a specific book"}
    - "Which books by Robert Martin do you have?" → {"is_db_query": true, "confidence": 0.95, "reasoning": "Asking about books by a specific author"}
    - "Are there any books on computer science still available?" → {"is_db_query": true, "confidence": 0.9, "reasoning": "Asking about book availability on a specific subject"}
    - "Is book 1948 available for renting?" → {"is_db_query": true, "confidence": 0.9, "reasoning": "Asking about book availability on a specific subject"}
    
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # OpenRouter model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        
        # Convert the string response to a Python dictionary
        import json
        parsed_response = json.loads(result)
        
        return (
            parsed_response["is_db_query"],
            parsed_response["confidence"],
            parsed_response["reasoning"]
        )
    except Exception as e:
        print(f"Error classifying library query intent: {e}")
        # Default to True (use DB) if there's an error
        return True, 0.6, f"Error during classification, defaulting to database query: {str(e)}"

def format_library_response(query: str, results: Dict[str, Any]) -> str:
    """
    Use an LLM to format the library query results into a well-structured, readable response.
    
    Args:
        query: The original user query
        results: The database query results
        
    Returns:
        A well-formatted, conversational response string
    """
    try:
        # Extract data from results
        data = results.get("results", [])
        sql_query = results.get("sql", "")
        
        # Prepare the system prompt
        system_prompt = """
        You are a library assistant that formats database query results into a conversational response.
        Use the provided query and SQL results to generate a friendly, informative response.
        """
        
        # Prepare the context with query results
        context = f"User query: {query}\n\nSQL query used: {sql_query}\n\nQuery results:\n"
        
        # Add result data
        if "error" in results:
            context += f"Error: {results['error']}"
        elif not data:
            context += "No books found matching the query."
        else:
            context += f"Found {len(data)} books:\n\n"
            for i, item in enumerate(data):
                context += f"Book {i+1}:\n"
                for key, value in item.items():
                    if value is not None:
                        context += f"- {key}: {value}\n"
                context += "\n"
        
        # Generate formatted response using OpenRouter
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # OpenRouter model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        formatted_response = response.choices[0].message.content
        return formatted_response
    
    except Exception as e:
        print(f"Error formatting library response: {e}")
        # Return a simple formatted response as fallback
        if "error" in results:
            return f"Sorry, I encountered an error while searching for books: {results['error']}"
        
        data = results.get("results", [])
        if not data:
            return "I couldn't find any books matching your query. Perhaps try different keywords or ask about another book?"
        
        response = f"Here are the books I found for '{query}':\n\n"
        for i, item in enumerate(data):
            response += f"**Book {i+1}**:\n"
            for key, value in item.items():
                if value is not None:
                    response += f"- **{key}**: {value}\n"
            response += "\n"
        
        return response

def process_library_query(query: str) -> Dict[str, Any]:
    """
    Process a query related to the library, determining if it should use the database
    endpoint or be handled by a general response using the RAG pipeline.
    
    Args:
        query: The user's question
        model: The LLM model to use for RAG pipeline (from Streamlit session state)
        
    Returns:
        A dictionary with the response and metadata
    """
    # Detect if this is a database query
    is_db_query, confidence, reasoning = detect_db_query_intent(query)
    
    # If it's a database query with reasonable confidence
    if is_db_query and confidence > 0.6:
        # Call the DB endpoint function
        results = db_endpoint(query)
        
        # Format the results using LLM
        formatted_response = format_library_response(query, results)
        print(f"Formatted response: {formatted_response}")
        
        return {
            "response": formatted_response,
            "is_db_query": True,
            "confidence": confidence,
            "reasoning": reasoning,
            "sql": results.get("sql"),
            "results": results
        }
    return None
