import streamlit as st
import requests
import time
import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import backend.metrics
import importlib
importlib.reload(backend.metrics)
from backend.metrics import metrics_tracker
from modules.shared.models import ModelManager
from rag.retriever import RAGRetriever

# Page config
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Professional Theme
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ecf0f1 !important;
        font-family: 'Inter', sans-serif;
    }
    p, div, li {
        color: #bdc3c7;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ecf0f1;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1rem;
        color: #95a5a6;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Chat Styling */
    .stChatMessage {
        background-color: #1e2530;
        border: 1px solid #2c3e50;
        border-radius: 10px;
    }
    
    /* Sidebar */
    .sidebar-content {
        background-color: #1e2530;
    }
    
    /* Disclaimer */
    .disclaimer {
        font-size: 0.8rem;
        color: #57606f;
        margin-top: 3rem;
        text-align: center;
        border-top: 1px solid #2c3e50;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# Sidebar Configuration
with st.sidebar:
    st.markdown("### Settings")
    
    # Model selection (Restricted to Meditron)
    st.info("Using Model: **Meditron (Medical LLM)**")
    model = "meditron"
    
    # RAG Toggle
    use_rag = st.toggle("Enable Medical Knowledge (RAG)", value=True)
    
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Interface
st.markdown('<div class="main-header">Medical Chatbot</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered medical Q&A with RAG</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)



# Chat Input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Analyzing medical knowledge base..."):
                # Initialize components if needed
                if 'model_manager' not in st.session_state:
                    st.session_state.model_manager = ModelManager()
                if 'rag_retriever' not in st.session_state:
                    st.session_state.rag_retriever = RAGRetriever()
                
                # 1. Retrieve Context
                context = ""
                citations = []
                if use_rag:
                    context, results = st.session_state.rag_retriever.retrieve(prompt)
                    citations = st.session_state.rag_retriever.get_citations(results)
                
                # 2. Generate Answer
                result = st.session_state.model_manager.generate(
                    model_name=model,
                    question=prompt,
                    context=context,
                    use_rag=use_rag
                )
                
                if not result.get('error'):
                    answer = result['answer']
                    
                    # Format answer
                    full_response = answer
                    
                    # Add citations if available
                    if citations:
                        full_response += "\n\n**Sources:**"
                        for i, cit in enumerate(citations[:3], 1):
                            source_text = f"\n> **[{i}] {cit['source']}**: {cit['answer'][:100]}..."
                            full_response += source_text
                    
                    # Display response
                    message_placeholder.markdown(full_response)
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Log metrics
                    metrics_tracker.log_inference("Medical Chatbot", 0.0, True, None) # Duration not tracked in this simplified view
                else:
                    error_msg = f"Error: {result.get('answer', 'Unknown error')}"
                    message_placeholder.error(error_msg)
                    metrics_tracker.log_inference("Medical Chatbot", 0.0, False, error_msg)

        except Exception as e:
            message_placeholder.error(f"Connection Error: {str(e)}")

# Feedback for the last assistant message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    # Initialize feedback state for the current message count (unique ID per response)
    msg_count = len(st.session_state.messages)
    feedback_key = f"feedback_submitted_chat_{msg_count}"
    
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = False

    st.markdown("### Rate the last response")
    
    if not st.session_state[feedback_key]:
        col_up, col_down = st.columns([1, 10])
        with col_up:
            if st.button("Helpful", key=f"like_chat_{msg_count}"):
                metrics_tracker.log_feedback("Medical Chatbot", True)
                st.session_state[feedback_key] = True
                st.rerun()
        with col_down:
            if st.button("Not Helpful", key=f"dislike_chat_{msg_count}"):
                metrics_tracker.log_feedback("Medical Chatbot", False)
                st.session_state[feedback_key] = True
                st.rerun()
    else:
        st.info("Thanks for your feedback!")

# Footer Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>Medical Disclaimer:</strong> This AI assistant is for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)
