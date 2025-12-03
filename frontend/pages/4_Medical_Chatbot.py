import streamlit as st
import requests
import time

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
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": prompt,
                        "model": model,
                        "use_rag": use_rag
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data['answer']
                    citations = data.get('citations', [])
                    
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
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    message_placeholder.error(error_msg)
        except Exception as e:
            message_placeholder.error(f"Connection Error: {str(e)}")

# Footer Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>Medical Disclaimer:</strong> This AI assistant is for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)
