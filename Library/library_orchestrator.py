import re
from openai import OpenAI
from typing import Dict, Any, Tuple, Optional
import os
from Library.DB_endpoint import db_endpoint

# Set up OpenRouter client for intent detection
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
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
    
    Examples:
    - "Do you have any books on machine learning?" → {"is_db_query": true, "confidence": 0.95, "reasoning": "Asking about specific books in the library database"}
    - "Is 'Clean Code' available in the library?" → {"is_db_query": true, "confidence": 0.9, "reasoning": "Asking about availability of a specific book"}
    - "Which books by Robert Martin do you have?" → {"is_db_query": true, "confidence": 0.95, "reasoning": "Asking about books by a specific author"}
    - "What are the library hours?" → {"is_db_query": false, "confidence": 0.8, "reasoning": "Asking about library operations, not database content"}
    - "How do I get a library card?" → {"is_db_query": false, "confidence": 0.9, "reasoning": "Asking about library services, not specific books"}
    - "Where is the science section in the library?" → {"is_db_query": false, "confidence": 0.85, "reasoning": "Asking about physical library layout, not database content"}
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

def process_library_query(query: str) -> Dict[str, Any]:
    """
    Process a query related to the library, determining if it should use the database
    endpoint or be handled by a general response (to be implemented later).
    
    Args:
        query: The user's question
        
    Returns:
        A dictionary with the response and metadata
    """
    # Detect if this is a database query
    is_db_query, confidence, reasoning = detect_db_query_intent(query)
    
    # If it's a database query with reasonable confidence
    if is_db_query and confidence > 0.6:
        # Call the DB endpoint function
        results = db_endpoint(query)
        
        # Format the results for display
        if "error" in results:
            response = f"Error processing library query: {results['error']}"
        else:
            response = f"Query: {results.get('query')}\n\n"
            
            data = results.get("results", [])
            if not data:
                response += "No books found matching your query."
            else:
                response += "Here are the matching books:\n\n"
                for i, item in enumerate(data):
                    response += f"**Book {i+1}:**\n"
                    for key, value in item.items():
                        if value is not None:  # Only show non-null values
                            response += f"- {key}: {value}\n"
                    response += "\n"
        
        return {
            "response": response,
            "is_db_query": True,
            "confidence": confidence,
            "reasoning": reasoning,
            "sql": results.get("sql"),
            "results": results
        }
    
    # For general library questions - placeholder for future implementation
    # For now, we'll still use the DB endpoint but with a note
    results = db_endpoint(query)
    
    if "error" in results:
        general_response = "I understand you're asking about general library information rather than specific books. " + \
                          "That functionality will be available soon. In the meantime, I'll try to find relevant books: \n\n" + \
                          f"Error processing query: {results['error']}"
    else:
        general_response = "I understand you're asking about general library information rather than specific books. " + \
                          "That functionality will be available soon. In the meantime, here are some possibly relevant books: \n\n"
        
        data = results.get("results", [])
        if not data:
            general_response += "No relevant books found."
        else:
            for i, item in enumerate(data):
                general_response += f"**Book {i+1}:**\n"
                for key, value in item.items():
                    if value is not None:  # Only show non-null values
                        general_response += f"- {key}: {value}\n"
                general_response += "\n"
    
    return {
        "response": general_response,
        "is_db_query": False,
        "confidence": confidence,
        "reasoning": reasoning,
        "sql": results.get("sql"),
        "results": results
    } 