import os
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_keywords_from_query(query: str) -> List[str]:
    """
    Extract potential keywords from a query by removing stopwords and short words.
    
    Args:
        query: The user's question or request
        
    Returns:
        A list of extracted keywords
    """
    # List of common stopwords to filter out
    stopwords = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                "in", "on", "at", "to", "for", "with", "by", "about", "like", "through",
                "over", "before", "after", "between", "under", "above", "of", "and", "or",
                "how", "what", "when", "where", "why", "who", "whom", "which", "there", "that"]
    
    # Convert to lowercase and tokenize by word boundaries
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Filter out stopwords and short words
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Look for specific academic terms
    academic_terms = ["admission", "tuition", "scholarship", "credit", "program", 
                     "degree", "major", "minor", "course", "class", "faculty", 
                     "semester", "exam", "library", "deadline", "graduate"]
    
    # Prioritize academic terms if found
    found_terms = [term for term in academic_terms if term in query.lower()]
    
    # Combine unique terms
    all_keywords = list(set(found_terms + keywords))
    
    return all_keywords

def extract_entities_from_query(query: str) -> Dict[str, List[str]]:
    """
    Extract specific entities like course codes, names, and dates from the query.
    
    Args:
        query: The user's question or request
        
    Returns:
        Dictionary of extracted entities by type
    """
    entities = {}
    
    # Extract course codes (e.g., CSC 226)
    course_pattern = r'\b([A-Z]{2,4})\s?(\d{3}[A-Z]?)\b'
    course_matches = re.findall(course_pattern, query)
    if course_matches:
        entities["course_codes"] = [f"{code[0]} {code[1]}" for code in course_matches]
        
    # Extract names (simple heuristic, not ideal)
    name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
    name_matches = re.findall(name_pattern, query)
    if name_matches:
        entities["names"] = name_matches
        
    # Extract dates
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    date_matches = re.findall(date_pattern, query)
    if date_matches:
        entities["dates"] = date_matches
    
    # Extract days of the week
    day_pattern = r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
    day_matches = re.findall(day_pattern, query, re.IGNORECASE)
    if day_matches:
        entities["days"] = [day.capitalize() for day in day_matches]
        
    # Extract time slots
    time_pattern = r'\b(\d{1,2}:\d{2}|\d{1,2})(am|pm|AM|PM)?\b'
    time_matches = re.findall(time_pattern, query)
    if time_matches:
        entities["times"] = [f"{t[0]}{t[1].lower()}" for t in time_matches if t[0]]
    
    return entities

def format_response(content: str, sources: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Format the final response with optional source citations.
    
    Args:
        content: The main response content
        sources: Optional list of sources used for the response
        
    Returns:
        Formatted response string
    """
    if not sources:
        return content
        
    # Format the response with source citations
    response = content + "\n\n**Sources:**\n"
    for i, source in enumerate(sources, 1):
        heading = source.get("heading", "Unknown")
        doc_id = source.get("id", f"source-{i}")
        response += f"{i}. {heading} (ID: {doc_id})\n"
    
    return response