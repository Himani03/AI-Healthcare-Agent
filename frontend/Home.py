import streamlit as st
from streamlit.components.v1 import html

# Page config
st.set_page_config(
    page_title="GenMedX: AI Healthcare Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .card {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-color: #1f77b4;
    }
    .icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .card-desc {
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 AI Healthcare Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Medical AI for Diagnosis & Risk Prediction</p>', unsafe_allow_html=True)

# Navigation Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="icon">💬</div>
        <div class="card-title">Medical Chatbot</div>
        <div class="card-desc">
            Interactive Q&A with 4 state-of-the-art LLMs (Llama 3, BioMistral, etc.) 
            enhanced by RAG for accurate medical information.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch Chatbot 🚀", use_container_width=True):
        st.switch_page("pages/1_Medical_Chatbot.py")

with col2:
    st.markdown("""
    <div class="card">
        <div class="icon">🩺</div>
        <div class="card-title">Risk Analysis</div>
        <div class="card-desc">
            AI-powered triage and risk prediction using BioMistral-7B. 
            Get clinical reasoning and test recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Analysis ⚡", use_container_width=True):
        st.switch_page("pages/2_Risk_Analysis.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; padding: 2rem;">
    <p>Powered by GenMedX • BioMistral • Llama 3 </p>
</div>
""", unsafe_allow_html=True)
