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

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return UniversityChatbot()

# Main app
def main():
    st.title("🎓 University Assistant")
    
    # Sidebar with language selection and search options
    with st.sidebar:
        st.subheader("Settings")
        language = st.selectbox("Language / اللغة", ["English", "Arabic"])
        
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
    
    chatbot = get_chatbot()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for new message
    if prompt := st.chat_input("Ask me a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.get_response(prompt, language=language)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 