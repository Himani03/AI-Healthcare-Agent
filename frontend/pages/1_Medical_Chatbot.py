"""
Professional Medical Chat Interface
"""
import streamlit as st
import requests
import time

# Page config
st.set_page_config(
    page_title="AI Healthcare Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical professional look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    /* Citation box styling */
    .citation-box {
        background-color: #f8f9fa;
        border-left: 3px solid #1f77b4;
        padding: 10px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #444;
    }
    
    /* Disclaimer styling */
    .disclaimer {
        font-size: 0.8rem;
        color: #7f8c8d;
        margin-top: 2rem;
        text-align: center;
        border-top: 1px solid #eee;
        padding-top: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# API URL
API_URL = "http://localhost:8000"

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model selection
    model_options = {
        "Llama 3 8B (Best)": "llama",
        "BioMistral 7B": "biomistral",
        "Gemini 1.5 Flash": "gemini",
        "Meditron 7B": "meditron"
    }
    
    selected_display = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0
    )
    model = model_options[selected_display]
    
    # RAG Toggle
    use_rag = st.toggle("Enable Medical Knowledge (RAG)", value=True)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Interface
st.markdown('<div class="main-header">🏥 AI Healthcare Agent</div>', unsafe_allow_html=True)

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
    <strong>⚠️ Medical Disclaimer:</strong> This AI assistant is for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)
