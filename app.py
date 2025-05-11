import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from utils.chatbot import UniversityChatbot

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
    "Anthropic Claude 3 Haiku($0.25/M)": "openrouter/anthropic/claude-3-haiku",
    "Anthropic Claude 3 Sonnet($3/M)": "openrouter/anthropic/claude-3-sonnet",
    "Anthropic Claude 3 Opus($15/M)": "openrouter/anthropic/claude-3-opus",
    "Google Gemini 2.0 Flash($0.10/M)": "openrouter/google/gemini-2.0-flash-001",
}

# Initialize chatbot when needed - not cached to avoid early initialization
def get_chatbot():
    if "chatbot" not in st.session_state:
        model = st.session_state.get("model")
        # Don't create the chatbot if no model is selected yet
        if not model:
            return None
        print(f"Creating new chatbot with model: {model}")
        st.session_state.chatbot = UniversityChatbot(model_name=model)
    return st.session_state.chatbot

# Main app
def main():
    st.title("🎓 University Assistant")
    
    # Sidebar with settings
    with st.sidebar:
        st.subheader("Settings")
        language = st.selectbox("Language / اللغة", ["English", "Arabic"])
        
        # Model selection dropdown - default to no selection
        model_options = ["-- Select a model --"] + list(AVAILABLE_MODELS.keys())
        selected_model_name = st.selectbox(
            "Select AI Model",
            model_options,
            index=0
        )
        
        model_selected = selected_model_name != "-- Select a model --"
        
        # Only set model if a valid selection was made
        if model_selected:
            # Get the model ID
            selected_model_id = AVAILABLE_MODELS[selected_model_name]
            
            # Update model if changed
            if "model" not in st.session_state or st.session_state.model != selected_model_id:
                # Remove previous chatbot if model changed
                if "chatbot" in st.session_state:
                    del st.session_state["chatbot"]
                st.session_state.model = selected_model_id
        else:
            # Clear model selection if "Select a model" is chosen
            if "model" in st.session_state:
                del st.session_state["model"]
                if "chatbot" in st.session_state:
                    del st.session_state["chatbot"]
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            if "chatbot" in st.session_state:
                st.session_state.chatbot.clear_history()
            st.rerun()
        
        # Advanced search options (expandable)
        with st.expander("Advanced Search Options"):
            st.write("These settings control how the assistant searches for information.")
            search_method = st.radio(
                "Search Method",
                ["Hybrid (Default)", "Vector Only", "Keyword Only"],
                index=0
            )
    
    st.subheader("University Information Assistant")
    st.write("Ask me anything about courses, schedules, exams, faculty, library resources, admission, or tuition!")
    
    # Initialize chat history in session state if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display model being used if selected
    if model_selected:
        st.caption(f"Currently using: {selected_model_name}")
    else:
        st.warning("Please select a model from the dropdown to begin chatting.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for new message - disable if no model selected
    prompt = st.chat_input("Ask me a question...", disabled=not model_selected)
    
    if prompt:
        # Get chatbot instance now that we need it
        chatbot = get_chatbot()
        
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
                response = chatbot.get_response(prompt, language=language)
                st.markdown(response)
        
        # Update message history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 