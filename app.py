import streamlit as st
import os
from dotenv import load_dotenv
import logging
import tempfile

# Import the base query classifier
from base.query_classifier import classify_query_sync

# Import RAG pipeline
from modules.classified_chatbot import rag_pipeline_simple

from Library.DB_endpoint import db_endpoint

# Import speech transcriber
from speech_to_text import SpeechTranscriber

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="University Assistant",
    page_icon="🎓",
    layout="wide"
)

# Define available models
AVAILABLE_MODELS = {
    "OpenAI GPT-4o Mini($0.15/M)": "gpt-4o-mini",
    "OpenAI GPT-4.1 Mini($0.40/M)": "gpt-4.1-mini",
    "OpenAI GPT-4.1-nano($0.10/M)": "gpt-4.1-nano",
    "Anthropic Claude 3 Haiku($0.25/M)": "claude-3-haiku",
    "Anthropic Claude 3 Sonnet($3/M)": "claude-3-sonnet",
    "Anthropic Claude 3 Opus($15/M)": "claude-3-opus",
    "Google Gemini 2.0 Flash($0.10/M)": "gemini-2.0-flash-001",
}

# Define which models should use OpenRouter instead of OpenAI API
OPENROUTER_MODELS = [
    "claude-3-haiku",
    "claude-3-sonnet",
    "claude-3-opus",
    "gemini-2.0-flash-001",
]

# Module map
MODULES = {
    "course_information": {
        "name": "Course Information",
        "collection_name": "admission_course_guide_json",
        "description": "Information about course content, prerequisites, credit hours, etc."
    },
    "class_schedules": {
        "name": "Class Schedules",
        "collection_name": "class_schedule_json",
        "description": "Details about when and where classes meet"
    },
    "exam_alerts": {
        "name": "Exam Data",
        "collection_name": "exam_data_json",
        "description": "Information about exam dates, deadlines, and assessments"
    },
    "study_resources": {
        "name": "Study Resources",
        "collection_name": "study_resource_json",
        "description": "Materials for studying including textbooks and online resources"
    },
    "professors": {
        "name": "Professors",
        "collection_name": "professor_data_json",
        "description": "Faculty information, office hours, and contact details"
    },
    "library": {
        "name": "Library",
        "description":"Information about available books and library resources"
    }
}

def get_module_response(query: str, language: str = "English") -> str:
    """
    Route the query to the appropriate module and get a response using RAG pipeline
    
    Args:
        query: The user's question
        language: The language for the response
        
    Returns:
        Formatted response string
    """
    try:
        # First, classify the query to determine which module should handle it
        classification = classify_query_sync(query)
        module_name = classification.module
        confidence = classification.confidence
        reasoning = classification.reasoning
        
        logger.info(f"Query classified as '{module_name}' with confidence {confidence}")
        
        # # # If confidence is below threshold, route to general response
      #   # if confidence < 0.3:
    #   #     logger.info(f"Low confidence ({confidence}), routing to generalesponse")
      #   #     returnet_general_response_sync(query, language)
        
        # Get the response function f the module
        # Special handling for library module
        if module_name == "library":
            logger.info("Using DB endpoint for library query")
            try:
                # Call the db_endpoint function with the user query
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
                
                # Add debug info if in development
                if os.getenv("APP_ENV") == "development":
                    response += f"\n\n---\nDebug: Query classified as '{module_name}' (confidence: {confidence:.2f})\nReasoning: {reasoning}\nSQL: {results.get('sql')}"
                
                return response
            except Exception as e:
                logger.error(f"Error in library DB endpoint: {e}", exc_info=True)
                return "Sorry, there was an error processing your library query."
        
        # For other modules, use the RAG pipeline
        # Determine which collection to use
        if module_name in MODULES:
            collection_name = MODULES[module_name]["collection_name"]
        else:
            # Fallback to study resources if module not found
            logger.warning(f"Module '{module_name}' not found, falling back to study resources")
            collection_name = MODULES["study_resources"]["collection_name"]
        
        # Modify query to include language preference
        language_prefix = ""
        if language.lower() == "arabic":
            language_prefix = "Please respond in Arabic: "
        
        modified_query = language_prefix + query
        
        # Get the selected model from session state
        model = st.session_state.get("model", "openai/gpt-4o-mini")
        
        # Check if we need to prepend the provider name for OpenRouter models
        if st.session_state.get("use_openrouter", False):
            # For OpenRouter models that aren't OpenAI, need to prepend provider
            if "gpt" not in model:
                if "claude" in model:
                    model = f"anthropic/{model}"
                elif "gemini" in model:
                    model = f"google/{model}"
            else:
                model = f"openai/{model}"
        
        # Use RAG pipeline to get response
        result = rag_pipeline_simple(modified_query, collection_name, model)
        response = result["response"]
        
        # Add debug info if in development
        debug_info = ""
        if os.getenv("APP_ENV") == "development":
            debug_info = f"\n\n---\nDebug: Query classified as '{module_name}' (confidence: {confidence:.2f})\nReasoning: {reasoning}"
        
        return response + debug_info
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        if language.lower() == "arabic":
            return "عذرًا، حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى لاحقًا."
        else:
            return "I'm sorry, an error occurred while processing your question. Please try again later."

# Initialize the speech transcriber
@st.cache_resource
def get_speech_transcriber():
    return SpeechTranscriber()

# Main app
def main():
    st.title("🎓 University Assistant")
    
    # Sidebar with settings
    with st.sidebar:
        st.subheader("Settings")
        language = st.selectbox("Language / اللغة", ["English", "Arabic"])
        
        # Model selection dropdown - default to first model
        model_options = list(AVAILABLE_MODELS.keys())
        selected_model_name = st.selectbox(
            "Select AI Model",
            model_options,
            index=0
        )
        
        selected_model_id = AVAILABLE_MODELS[selected_model_name]
        
        # Store model ID and whether it's an OpenRouter model in session state
        if "model" not in st.session_state or st.session_state.model != selected_model_id:
            st.session_state.model = selected_model_id
            st.session_state.use_openrouter = selected_model_id in OPENROUTER_MODELS
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()
        
        # Show module information
        st.subheader("Available Modules")
        for module_id, module_info in MODULES.items():
            with st.expander(module_info["name"]):
                st.write(module_info["description"])
    
    st.subheader("University Information Assistant")
    st.write("Ask me anything about courses, schedules, exams, faculty, library resources, admission, or tuition!")
    
    # Display model being used
    st.caption(f"Currently using: {selected_model_name}")
    
    # Initialize chat history in session state if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Create columns for text and speech input
    col1, col2 = st.columns([5, 1])
    
    # Text input
    with col1:
        prompt = st.chat_input("Ask me a question...")
    
    # Speech input
    with col2:
        speech_button = st.button("🎤 Record")
    
    # Handle speech input
    if speech_button:
        # Get the transcriber
        transcriber = get_speech_transcriber()
        
        # Show recording status
        with st.spinner("Recording..."):
            try:
                # Record and transcribe
                prompt = transcriber.listen_and_transcribe(duration=7)
                st.success(f"Transcribed: {prompt}")
            except Exception as e:
                st.error(f"Error recording audio: {str(e)}")
                prompt = None
    
    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add to session state message history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model_name}..."):
                response = get_module_response(prompt, language=language)
                st.markdown(response)
        
        # Update message history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()