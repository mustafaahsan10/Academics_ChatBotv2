import os
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict
import pydantic_ai
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import re
from qdrant_client import QdrantClient

# Import our custom Agent SDK template
from agent_sdk_template import Agent as LangGraphAgent
from agent_sdk_template import AgentConfig, Tool, ToolParameter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "course_information"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Pydantic models for course information
# class CourseQueryContext(BaseModel):
#     """Context for a course information query"""
#     user_query: str = Field(..., description="The user's original query about course information")
#     processed_query: str = Field("", description="The processed query with stopwords removed")
#     keywords_matches: List[Dict] = Field(default_factory=list, description="Matches found in the keywords field")
#     heading_matches: List[Dict] = Field(default_factory=list, description="Matches found in the heading field")
#     context: Optional[str] = Field(None, description="Context information retrieved from search")
#     response_language: str = Field("English", description="Language to respond in (English or Arabic)")

class CourseSearchRequest(BaseModel):
    """Request parameters for searching course information"""
    query: str = Field(..., description="The query to search for course information")
    language: str = Field("English", description="The language to respond in")

class CourseSearchResult(BaseModel):
    """Result from searching course information"""
    context: str = Field(..., description="Context information retrieved from search")

class CourseResponse(BaseModel):
    """Structured response for course information queries"""
    answer: str = Field(..., description="Comprehensive answer addressing the user's query about courses")

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

def extract_terms_of_interest(query: str) -> List[str]:
    """Extract important academic terms from the query"""
    academic_terms = [
        "major", "majors", "program", "programs", "degree", "degrees", "faculty", "faculties",
        "undergraduate", "graduate", "credits", "credit", "course", "courses", "prerequisite",
        "prerequisites", "requirement", "requirements", "engineering", "business", "science",
        "arts", "humanities", "law", "nursing", "health", "computer", "architecture"
    ]
    
    # Look for these terms in the query
    found_terms = []
    for term in academic_terms:
        if term in query.lower().split():
            found_terms.append(term)
    
    return found_terms

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
        terms_of_interest = extract_terms_of_interest(query)
        
        logger.info(f"Processed query: '{processed_query}', terms: {terms_of_interest}")
        
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
        # We'll examine the results and boost those with keyword/heading matches
        enhanced_results = []
        
        for result in initial_results:
            payload = result.payload
            metadata = payload.get("metadata", {})
            
            # Initialize score components
            base_score = result.score
            keyword_boost = 0.0
            heading_boost = 0.0
            course_details_boost = 0.0
            prerequisite_boost = 0.0
            
            # 1. Check for matches in keywords
            keywords = metadata.get("keywords", [])
            if isinstance(keywords, list) and keywords:
                # Look for matches between query terms and keywords
                for term in processed_query.split():
                    # Check if term appears in any keyword
                    if any(term.lower() in kw.lower() for kw in keywords):
                        keyword_boost += 0.1
                
                # Extra boost for terms of interest
                for term in terms_of_interest:
                    if any(term.lower() in kw.lower() for kw in keywords):
                        keyword_boost += 0.2
            
            # 2. Check for matches in heading
            heading = metadata.get("heading", "")
            if heading:
                # Look for matches between query terms and heading
                heading_lower = heading.lower()
                for term in processed_query.split():
                    if term.lower() in heading_lower:
                        heading_boost += 0.15
                
                # Extra boost for terms of interest in heading
                for term in terms_of_interest:
                    if term.lower() in heading_lower:
                        heading_boost += 0.25
            
            # 3. Check for matches in course details
            course_details = metadata.get("course_details", {})
            if course_details:
                # Check for course code matches (higher priority)
                course_code = course_details.get("code", "")
                if course_code and any(term.lower() in course_code.lower() for term in processed_query.split()):
                    course_details_boost += 0.3
                
                # Check for course name matches
                course_name = course_details.get("name", "")
                if course_name and any(term.lower() in course_name.lower() for term in processed_query.split()):
                    course_details_boost += 0.2
                
                # Check for matches in description
                description = course_details.get("description", "")
                if description and any(term.lower() in description.lower() for term in processed_query.split()):
                    course_details_boost += 0.1
            
            # 4. Check for matches in prerequisites
            prerequisites = metadata.get("prerequisite_details", [])
            if isinstance(prerequisites, list) and prerequisites:
                # Look for matches in prerequisites
                if any(any(term.lower() in prereq.lower() for term in processed_query.split()) for prereq in prerequisites):
                    prerequisite_boost += 0.15
            
            # 5. Calculate final score with all boosts
            final_score = base_score * (1.0 + keyword_boost + heading_boost + course_details_boost + prerequisite_boost)
            
            # Create enhanced result with new score
            enhanced_result = {
                "id": result.id,
                "payload": payload,
                "score": final_score,
                "boosts": {
                    "keyword_boost": keyword_boost,
                    "heading_boost": heading_boost,
                    "course_details_boost": course_details_boost,
                    "prerequisite_boost": prerequisite_boost
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

def format_search_results(results) -> str:
    """Format search results into a context string"""
    if not results:
        return "No specific information found related to your query."
    
    context_parts = []
    for i, hit in enumerate(results):
        payload = hit.payload
        text = payload.get("text", "")
        metadata = payload.get("metadata", {})
        
        # Format heading info
        heading_info = ""
        if metadata.get("heading"):
            heading_info = f"Section: {metadata['heading']}\n"
        
        # Format keywords for context
        keywords_info = ""
        if metadata.get("keywords") and isinstance(metadata.get("keywords"), list):
            keywords = metadata.get("keywords")
            keywords_info = f"Keywords: {', '.join(keywords)}\n"
        
        # Format course details if available
        course_details_info = ""
        if metadata.get("course_details"):
            course_details = metadata.get("course_details")
            if course_details:
                if course_details.get("code"):
                    course_details_info += f"Course Code: {course_details['code']}\n"
                if course_details.get("name"):
                    course_details_info += f"Course Name: {course_details['name']}\n"
                if course_details.get("credits"):
                    course_details_info += f"Credits: {course_details['credits']}\n"
                if course_details.get("description"):
                    course_details_info += f"Description: {course_details['description']}\n"
        
        # Format prerequisite details if available
        prereq_info = ""
        if metadata.get("prerequisite_details") and isinstance(metadata.get("prerequisite_details"), list):
            prerequisites = metadata.get("prerequisite_details")
            if prerequisites:
                prereq_info = f"Prerequisites: {', '.join(prerequisites)}\n"
        
        # Format content with score
        context_parts.append(f"{heading_info}{keywords_info}{course_details_info}{prereq_info}\nContent: {text}\n(Relevance Score: {hit.score:.2f})")
    
    return "\n\n---\n\n".join(context_parts)

# Create a tool for course information search
def search_course_info(request: CourseSearchRequest) -> CourseSearchResult:
    """
    Search for course information based on the query
    
    Args:
        request: The search request with query and language
        
    Returns:
        Results matching the search query
    """
    try:
        # Perform hierarchical search - return all results
        search_results = hierarchical_search(request.query, top_k=None)
        
        # Format the results
        context = format_search_results(search_results)
        
        return CourseSearchResult(
            context=context
        )
            
    except Exception as e:
        logger.error(f"Error searching course information: {e}")
        return CourseSearchResult(
            context="Error searching for course information."
        )

# Function to create a LangGraph-based agent with OpenRouter
def create_langgraph_agent():
    """
    Create an agent using our custom LangGraph-based Agent SDK template with OpenRouter
    """
    # Get model from Streamlit session state
    model_id = "gpt-4o-mini"  # Default model
    if "model" in st.session_state:
        model_id = st.session_state.model

    # Create our search tool using the LangGraph Tool format
    search_tool = Tool(
        name="search_course_info",
        description="Search for course information based on the query",
        parameters=[
            ToolParameter(
                name="query",
                description="The query to search for course information",
                type="string",
                required=True
            ),
            ToolParameter(
                name="language",
                description="The language to respond in (English or Arabic)",
                type="string",
                required=False,
                default="English"
            )
        ],
        function=lambda query, language="English": search_course_info(
            CourseSearchRequest(query=query, language=language)
        ).context
    )
    
    # System instructions for the agent
    system_instructions = """You are a knowledgeable university assistant specializing in course information.
Your goal is to provide accurate, helpful information about courses, their content, programs, majors, 
and academic requirements.

You have access to a search_course_info tool that can find information about university courses, degree programs, 
academic requirements, and available majors. Always use this tool to look up information before answering questions.

When responding to queries:
1. Be specific about program names and degree requirements
2. Include important information found in the search results
3. Format your answers in a clear, structured way

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about university courses and programs. Never say phrases like "I don't have information on this" or "this isn't in my database". Always maintain a helpful tone and try to address the user's query as best as possible.

IMPORTANT: Respond in English by default. Only respond in Arabic if explicitly instructed to do so with "(Please respond in Arabic)" in the query.

Remember that students rely on your accuracy for their academic planning, so be specific and clear.
"""
    
    # Create agent configuration
    config = AgentConfig(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model=model_id,
        temperature=0.7,
        tools=[search_tool],
        instructions=system_instructions
    )
    
    # Create and return the agent
    return LangGraphAgent(config)

# Function to get an AI agent with the dynamically selected model
def get_course_agent():
    """
    Get the AI agent for course information with the model selected by the user
    """
    # Get model from Streamlit session state if available, otherwise use default
    model_id = "gpt-4o-mini"  # Default model
    use_openrouter = False
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if "model" in st.session_state:
        model_id = st.session_state.model
        # Check if we should use OpenRouter based on session state
        if "use_openrouter" in st.session_state:
            use_openrouter = st.session_state.use_openrouter
    
    logger.info(f"Using model: {model_id} for course information with tools")
    
    # If using OpenRouter, return our custom LangGraph agent
    if use_openrouter:
        logger.info(f"Using LangGraph agent with OpenRouter for model: {model_id}")
        return None  # We'll handle this separately
    else:
        # For standard OpenAI models, use pydantic_ai
        logger.info(f"Using pydantic_ai agent with OpenAI for model: {model_id}")
        
        # System prompt for pydantic_ai agent
        system_prompt = """You are a knowledgeable university assistant specializing in course information.
Your goal is to provide accurate, helpful information about courses, their content, programs, majors, 
and academic requirements.

You have access to a search_course_info tool that can find information about university courses, degree programs, 
academic requirements, and available majors. Always use this tool to look up information before answering questions.

When responding to queries:
1. Be specific about program names and degree requirements
2. Include important information found in the search results
3. Format your answers in a clear, structured way

IMPORTANT: If the search results don't contain relevant information or if no information was found, DO NOT mention this limitation to the user. Instead, provide a helpful general response based on common knowledge about university courses and programs. Never say phrases like "I don't have information on this" or "this isn't in my database". Always maintain a helpful tone and try to address the user's query as best as possible.

IMPORTANT: Respond in English by default. Only respond in Arabic if explicitly instructed to do so with "(Please respond in Arabic)" in the query.

Remember that students rely on your accuracy for their academic planning, so be specific and clear.
"""

        # Create the agent with tools for OpenAI
        api_config = {
            "model": model_id,
            "tools": [search_course_info],
            "system_prompt": system_prompt,
            "openai_api_key": openai_api_key
        }
        
        # Create the pydantic_ai agent with the appropriate config
        course_agent = pydantic_ai.Agent(**api_config)
        return course_agent

async def get_course_response(query: str, language: str = "English") -> str:
    """
    Get a response for a course information query
    
    Args:
        query: The user's question about courses
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
                
        logger.info(f"Processing query with model: {model_id}, using OpenRouter: {use_openrouter}, language: {language}")
        
        # Prepare user message with language preference
        user_message = query
        if language.lower() == "arabic":
            user_message = f"{query} (Please respond in Arabic)"
        else:
            # Explicitly request English response
            user_message = f"{query} (Please respond in English)"
        
        if use_openrouter:
            # Use our LangGraph-based Agent with OpenRouter
            logger.info(f"Using LangGraph agent with OpenRouter for model: {model_id}")
            langgraph_agent = create_langgraph_agent()
            
            # Run the agent and get the response
            result = langgraph_agent.run(user_message)
            response = result["response"]
            logger.info(f"Received LangGraph agent response")
            return response
        else:
            # Use pydantic_ai agent for OpenAI models
            logger.info(f"Using pydantic_ai agent with model: {model_id}")
            course_agent = get_course_agent()
            response = await course_agent.run(user_message)
            logger.info(f"Received response type: {type(response)}")
            return response.output
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating course response: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة استفسارك عن المقرر الدراسي. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your course information query. Please try again."

def get_course_response_sync(query: str, language: str = "English") -> str:
    """
    Synchronous wrapper for get_course_response
    
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
        
    return loop.run_until_complete(get_course_response(query, language))