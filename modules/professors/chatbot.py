import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
import pydantic_ai
from pydantic import BaseModel, Field
import re
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Collection name for this module
COLLECTION_NAME = "professors"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Define models for the pydantic AI agent
# class ProfessorQueryContext(BaseModel):
#     """Simplified context for professor queries"""
#     user_query: str = Field(..., description="The user's original query about professors")
#     professor_name: Optional[str] = Field(None, description="Professor name mentioned in the query")
#     course_name: Optional[str] = Field(None, description="Course name mentioned in the query")
#     keywords: List[str] = Field(default_factory=list, description="Important keywords extracted from the query")
#     search_results: Optional[List[Dict[Any, Any]]] = Field(None, description="Relevant professor information retrieved")
#     response_language: str = Field("English", description="Language to respond in (English or Arabic)")

# Define models for the search tool
class ProfessorSearchRequest(BaseModel):
    """Request parameters for searching professor information"""
    query: str = Field(..., description="The query to search for professor information")
    language: str = Field("English", description="The language to respond in")

class ProfessorSearchResult(BaseModel):
    """Result from searching professor information"""
    context: str = Field(..., description="Context information retrieved from search")

# Define output model for the agent
class ProfessorResponse(BaseModel):
    """Response to a professor information query"""
    answer: str = Field(..., description="Detailed answer to the user's question about professors")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions related to professors")

# Initialize embeddings and Qdrant client
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Helper functions for search
def process_query(query: str) -> str:
    """Process the query to remove stopwords and prepare for search"""
    stopwords = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "in", 
                 "on", "at", "to", "for", "with", "by", "about", "like", "how", 
                 "what", "when", "where", "why", "who", "which", "can", "you", "tell",
                 "me", "list", "all", "available", "have", "does"]
    
    # Convert to lowercase and tokenize
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Remove stopwords and short words
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Join back into a string
    processed_query = " ".join(keywords)
    
    logger.info(f"Processed query: '{processed_query}' from original: '{query}'")
    return processed_query

def extract_professor_entities(query: str) -> Dict[str, Any]:
    """Extract professor-related entities from the query"""
    entities = {
        "professor_name": None,
        "course_name": None,
        "keywords": []
    }
    
    # Look for professor name pattern (e.g., "Professor Smith", "Dr. Johnson")
    professor_patterns = [
        r'professor\s+([A-Za-z\s\.]+?)(?:\s|$|\'s|\?|\.)',
        r'prof\.\s+([A-Za-z\s\.]+?)(?:\s|$|\'s|\?|\.)',
        r'dr\.\s+([A-Za-z\s\.]+?)(?:\s|$|\'s|\?|\.)'
    ]
    
    for pattern in professor_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            entities["professor_name"] = matches[0].strip()
            entities["keywords"].append(entities["professor_name"])
            break
    
    # Check for specific professor names mentioned without titles
    common_professors = ["Hoda Maalouf", "Nazir Hawi"]
    for name in common_professors:
        if name.lower() in query.lower():
            entities["professor_name"] = name
            entities["keywords"].append(name)
    
    # Look for course names or codes - simplified pattern
    course_patterns = [
        r'(?:course|class|teach(?:es|ing)?)\s+([A-Za-z0-9\s\-]+?)(?:\s|$|\?|\.)',
        r'([A-Z]{2,4}\s*\d{3}[A-Za-z0-9]*)'  # Course codes like CSC 226
    ]
    
    for pattern in course_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            entities["course_name"] = matches[0].strip()
            entities["keywords"].append(entities["course_name"])
            break
    
    # Add professor-related keywords
    professor_keywords = ["contact", "email", "phone", "office", "hours", "research", "publication", "biography", "teaching"]
    query_words = query.lower().split()
    for keyword in professor_keywords:
        if keyword in query_words:
            entities["keywords"].append(keyword)
    
    logger.info(f"Extracted entities: {entities}")
    return entities

def hierarchical_search(query: str, top_k: int = None) -> List[Dict]:
    """
    Perform hierarchical search by first using vector search and then
    post-processing to prioritize results with matching metadata
    
    Args:
        query: The search query
        top_k: Maximum number of results to return (None = return all results)
        
    Returns:
        List of search results with enhanced scoring
    """
    try:
        logger.info(f"Starting hierarchical search for query: {query}")
        
        # Process the query
        processed_query = process_query(query)
        entities = extract_professor_entities(query)
        
        # Get query embedding for vector search
        query_embedding = embeddings.embed_query(processed_query)
        
        # Get a larger initial result set for reranking
        initial_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=100  # Increased from 25 to get more potential matches
        )
        
        logger.info(f"Found {len(initial_results)} results in vector search")
        
        if not initial_results:
            logger.info("No results found")
            return []
        
        # Post-process results to implement hierarchical search
        enhanced_results = []
        
        for result in initial_results:
            payload = result.payload
            
            # Initialize score components
            base_score = result.score
            professor_name_boost = 0.0
            course_name_boost = 0.0
            heading_boost = 0.0
            keyword_boost = 0.0
            
            # 1. Check for professor name match
            if entities["professor_name"] and "professor_name" in payload:
                if entities["professor_name"].lower() in payload["professor_name"].lower():
                    professor_name_boost = 0.5  # Strong boost for exact professor name match
            
            # Also check for professor name in heading
            if entities["professor_name"] and "heading" in payload:
                if entities["professor_name"].lower() in payload["heading"].lower():
                    professor_name_boost = 0.4
            
            # 2. Check for course name match
            if entities["course_name"] and "text" in payload:
                if entities["course_name"].lower() in payload["text"].lower():
                    course_name_boost = 0.3
            
            # 3. Check for heading matches with query keywords
            if "heading" in payload:
                heading_lower = payload["heading"].lower()
                
                # Look for important sections like "contact information", "office hours", etc.
                if "contact" in processed_query and "contact" in heading_lower:
                    heading_boost += 0.4
                elif "hour" in processed_query and "hour" in heading_lower:
                    heading_boost += 0.4
                elif "office" in processed_query and "office" in heading_lower:
                    heading_boost += 0.4
                elif "biography" in processed_query and "biography" in heading_lower:
                    heading_boost += 0.4
                elif "research" in processed_query and "research" in heading_lower:
                    heading_boost += 0.3
                elif "publication" in processed_query and "publication" in heading_lower:
                    heading_boost += 0.3
                elif "journal" in processed_query and "journal" in heading_lower:
                    heading_boost += 0.3
                
                # General heading match
                for term in processed_query.split():
                    if term.lower() in heading_lower:
                        heading_boost += 0.1
            
            # 4. Check for keywords match
            if "keywords" in payload and isinstance(payload["keywords"], list):
                for keyword in processed_query.split():
                    if any(keyword.lower() in kw.lower() for kw in payload["keywords"]):
                        keyword_boost += 0.05
            
            # Calculate final score with boosts
            final_score = base_score * (1.0 + professor_name_boost + course_name_boost + 
                                       heading_boost + keyword_boost)
            
            # Create enhanced result with new score
            enhanced_result = {
                "id": result.id,
                "payload": payload,
                "score": final_score,
                "boosts": {
                    "professor_name_boost": professor_name_boost,
                    "course_name_boost": course_name_boost,
                    "heading_boost": heading_boost,
                    "keyword_boost": keyword_boost
                }
            }
            
            enhanced_results.append(enhanced_result)
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k results or all results if top_k is None
        if top_k is not None:
            top_results = enhanced_results[:top_k]
        else:
            top_results = enhanced_results
        
        # Convert back to format expected by the rest of the code
        final_results = []
        for result in top_results:
            # Create a ScoredPoint-like object
            class ScoredPoint:
                def __init__(self, id, payload, score):
                    self.id = id
                    self.payload = payload
                    self.score = score
            
            scored_point = ScoredPoint(
                id=result["id"],
                payload=result["payload"],
                score=result["score"]
            )
            
            final_results.append(scored_point)
        
        logger.info(f"Returned {len(final_results)} results after hierarchical boost")
        return final_results
        
    except Exception as e:
        logger.error(f"Error in hierarchical search: {e}")
        # Fall back to basic vector search
        try:
            logger.info("Attempting fallback vector search...")
            limit = 100 if top_k is None else top_k
            fallback_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit
            )
            logger.info(f"Fallback search found {len(fallback_results)} results")
            return fallback_results
        except Exception as e2:
            logger.error(f"Fallback search failed: {e2}")
            return []

def format_professor_results(results) -> str:
    """Format search results into a context string"""
    if not results:
        return "No specific information found related to your query."
    
    context_parts = []
    for i, hit in enumerate(results):
        payload = hit.payload
        
        # Format the professor information in a readable way
        prof_info = ""
        
        # Add heading information
        if "heading" in payload:
            prof_info += f"Section: {payload['heading']}\n\n"
        
        # Add text content
        if "text" in payload:
            prof_info += f"{payload['text']}\n\n"
        
        # Add keywords if available
        if "keywords" in payload and isinstance(payload["keywords"], list):
            keywords = payload["keywords"]
            prof_info += f"Keywords: {', '.join(keywords)}\n"
        
        # Add document source if available
        if "document_source" in payload:
            prof_info += f"Source: {payload['document_source']}\n"
        
        # Add relevance score
        prof_info += f"(Relevance Score: {hit.score:.2f})"
        
        context_parts.append(prof_info)
    
    return "\n\n---\n\n".join(context_parts)

# Create a tool for professor information search
def search_professor_info(request: ProfessorSearchRequest) -> ProfessorSearchResult:
    """
    Search for professor information based on the query
    
    Args:
        request: The search request with query and language
        
    Returns:
        Results matching the search query
    """
    try:
        # Perform hierarchical search - return all results
        search_results = hierarchical_search(request.query, top_k=None)
        
        # Format the results
        context = format_professor_results(search_results)
        
        return ProfessorSearchResult(
            context=context
        )
            
    except Exception as e:
        logger.error(f"Error searching professor information: {e}")
        return ProfessorSearchResult(
            context="Error searching for professor information."
        )

# Function to get an AI agent with the dynamically selected model
def get_professor_agent():
    """
    Get the AI agent for professor information with the model selected by the user
    """
    # Get model from Streamlit session state if available, otherwise use default
    model_id = "gpt-4o-mini"  # Default model
    use_openrouter = False
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if "model" in st.session_state:
        model_id = st.session_state.model
        # Check if we should use OpenRouter
        if "use_openrouter" in st.session_state and st.session_state.use_openrouter:
            use_openrouter = True
    
    logger.info(f"Using model: {model_id} for professor information with tools")
    if use_openrouter:
        logger.info(f"Using OpenRouter API for model: {model_id}")
    
    # Create the agent with tools
    professor_agent = pydantic_ai.Agent(
        model=model_id,
        tools=[search_professor_info],
        system_prompt="""You are a specialized university faculty information assistant.
Your goal is to provide accurate information about professors, their courses, and contact details.

You have access to a search_professor_info tool that can find information about faculty members,
their contact details, office hours, biographies, research interests, and publications.
Always use this tool to look up information before answering questions.

When responding:
1. Be specific about professor names and titles
2. Include course codes when mentioning which courses they teach
3. Provide clear and accurate information based on the search results

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about university professors and faculty. Never say phrases like "I don't have information on this" or "this isn't in my database". Always maintain a helpful tone and try to address the user's query as best as possible.

For Arabic queries, respond in fluent Arabic. For English queries, respond in clear English.

Remember that students rely on your accuracy when seeking information about faculty.
""",
        openai_api_key=openrouter_api_key if use_openrouter else openai_api_key,
        openai_api_base="https://openrouter.ai/api/v1" if use_openrouter else None
    )
    return professor_agent

async def get_professor_response(query: str, language: str = "English") -> str:
    """
    Get a response to a professor information query
    
    Args:
        query: The user's question about professors
        language: The language for the response
        
    Returns:
        Formatted response string
    """
    try:
        # Check if OpenRouter should be used
        use_openrouter = False
        model_id = "gpt-4o-mini"  # Default model
        
        if "model" in st.session_state:
            model_id = st.session_state.model
            if "use_openrouter" in st.session_state:
                use_openrouter = st.session_state.use_openrouter
                
        logger.info(f"Processing query with model: {model_id}, using OpenRouter: {use_openrouter}")
        
        # Prepare user message with language preference
        user_message = query
        if language.lower() == "arabic":
            user_message = f"{query} (Please respond in Arabic)"
        
        if use_openrouter:
            # Use direct OpenRouter API call
            import requests
            import json
            
            # Get the search results first
            search_request = ProfessorSearchRequest(query=query, language=language)
            search_result = search_professor_info(search_request)
            context = search_result.context
            
            # Prepare system message with context
            system_message = """You are a specialized university faculty information assistant.
Your goal is to provide accurate information about professors, their courses, and contact details
based on the following information:

""" + context + """

When responding:
1. Be specific about professor names and titles
2. Include course codes when mentioning which courses they teach
3. Provide clear and accurate information based on the search results

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about university professors and faculty. Never say phrases like "I don't have information on this" or "this isn't in my database". Always maintain a helpful tone and try to address the user's query as best as possible.

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
                "HTTP-Referer": "https://university-chatbot.com",  # Replace with your actual site URL
                "X-Title": "University Chatbot"  # Replace with your actual site name
            }
            
            # Format the model name for OpenRouter
            # Claude models need "anthropic/" prefix, and Gemini needs "google/" prefix
            if model_id == "claude-3-haiku" or model_id == "claude-3-sonnet" or model_id == "claude-3-opus":
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
        professor_agent = get_professor_agent()
        response = await professor_agent.run(user_message)
        
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
        logger.error(f"Error generating professor response: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة استفسارك عن أعضاء هيئة التدريس. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your faculty information query. Please try again."

def get_professor_response_sync(query: str, language: str = "English") -> str:
    """
    Synchronous wrapper for get_professor_response
    
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
        
    return loop.run_until_complete(get_professor_response(query, language))