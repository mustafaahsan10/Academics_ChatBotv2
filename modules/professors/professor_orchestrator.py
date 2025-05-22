from openai import OpenAI
from typing import Dict, Any, Tuple, Optional
import os
# Set up OpenAI client for intent detection

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def detect_meeting_intent(query: str) -> Tuple[bool, float, str]:
    """
    Detect if the query is about scheduling a meeting with a professor.
    
    Args:
        query: The user's question
        
    Returns:
        Tuple containing:
        - Boolean indicating if the intent is to schedule a meeting
        - Confidence score (0-1)
        - Reasoning for the classification
    """
    system_prompt = """
    You are an intent classifier for a university assistant. Determine if the query is about scheduling a meeting with a professor.
    
    Return ONLY a JSON object with the following structure:
    {
        "is_meeting_intent": true/false,
        "confidence": <float between 0 and 1>,
        "reasoning": "<brief explanation>",
        "professor_name": "<name of the professor>"
    }
    
    Examples of MEETING REQUESTS (return true):
    - "I want to schedule a meeting with Professor Smith" → {"is_meeting_intent": true, "confidence": 0.95, "reasoning": "Explicitly mentions scheduling a meeting with a professor"}
    - "Can I book an appointment with Dr. Johnson?" → {"is_meeting_intent": true, "confidence": 0.9, "reasoning": "Asks about booking an appointment with a professor"}
    - "How do I set up a time to meet with Professor Williams?" → {"is_meeting_intent": true, "confidence": 0.9, "reasoning": "Asking about setting up meeting time"}
    
    Examples of GENERAL QUERIES (return false):
    - "What are Professor Davis's office hours?" → {"is_meeting_intent": false, "confidence": 0.8, "reasoning": "Asking about office hours, not scheduling a meeting"}
    - "Tell me about Professor Wilson's research interests" → {"is_meeting_intent": false, "confidence": 0.95, "reasoning": "Asking about research interests, not scheduling"}
    - "What courses does Dr. Garcia teach?" → {"is_meeting_intent": false, "confidence": 0.95, "reasoning": "Asking about courses taught, not scheduling"}
    - "Can you tell me Professor Brown's email address?" → {"is_meeting_intent": false, "confidence": 0.9, "reasoning": "Asking for contact information, not scheduling"}
    - "What is Professor Lee's background?" → {"is_meeting_intent": false, "confidence": 0.95, "reasoning": "Asking about professor's background, not scheduling"}
    - "Where is Dr. Taylor's office located?" → {"is_meeting_intent": false, "confidence": 0.9, "reasoning": "Asking about office location, not scheduling"}
    
    Only classify as a meeting intent if the user is specifically asking to schedule, book, or arrange a meeting/appointment with a professor. General information queries about professors should be classified as false.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
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
            parsed_response["is_meeting_intent"],
            parsed_response["confidence"],
            parsed_response["reasoning"],
            parsed_response["professor_name"]
        )
    except Exception as e:
        print(f"Error classifying meeting intent: {e}")
        # Default to False if there's an error
        return False, 0.0, f"Error during classification: {str(e)}"

def generate_meeting_link(professor_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a meeting link for scheduling with a professor.
    This is a placeholder that will be replaced with Calendly integration later.
    
    Args:
        professor_name: Optional name of the professor
        
    Returns:
        Dictionary with meeting link information
    """
    # Extract professor name from the query if provided
    professor_display = f" with {professor_name}" if professor_name else ""
    
    # Placeholder demo link - this would be replaced with Calendly in the future
    demo_link = "https://university-demo.com/schedule-meeting"
    if professor_name:
        # Clean the professor name for URL (remove titles and simplify)
        clean_name = professor_name.lower().replace("professor ", "").replace("dr. ", "").replace(" ", "-")
        demo_link += f"/{clean_name}"
    
    return {
        "meeting_link": demo_link,
        "professor": professor_name,
        "instructions": f"Click the link below to schedule a meeting{professor_display}. "
                       f"This is a demo link that will be replaced with a real scheduling system in the future."
    }

def process_professor_query(query: str, collection_name: str) -> Dict[str, Any]:
    """
    Process a query related to professors, determining if it's about scheduling
    a meeting or a general question.
    
    Args:
        query: The user's question
        collection_name: The name of the Qdrant collection for professors
        
    Returns:
        A dictionary with the response and metadata
    """
    # Detect if this is a meeting scheduling intent
    is_meeting, confidence, reasoning, professor_name = detect_meeting_intent(query)
    
    # If it's a meeting request with reasonable confidence
    if is_meeting and confidence > 0.7:
        # Extract professor name if possible
        # professor_name = extract_professor_name(query)
        
        # Generate meeting link
        meeting_info = generate_meeting_link(professor_name)
        
        response = f"I'd be happy to help you schedule a meeting{' with ' + professor_name if professor_name else ''}. " + meeting_info["instructions"]
        response += f"\n\nScheduling Link: {meeting_info['meeting_link']}"
        
        return {
            "response": response,
            "meeting_info": meeting_info,
            "is_meeting_request": True,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    # For generic questions, return None to indicate the main RAG pipeline should be used
    return None 