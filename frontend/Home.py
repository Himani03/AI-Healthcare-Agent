import streamlit as st
from streamlit.components.v1 import html
import base64
import os
import sys

# Add root directory to path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Function to load image as base64
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load Images
try:
    img_main = get_img_as_base64("frontend/assets/logo_main.jpg")
    img_chatbot = get_img_as_base64("frontend/assets/logo_chatbot.jpg")
    img_risk = get_img_as_base64("frontend/assets/logo_risk.jpg")
    img_symptom = get_img_as_base64("frontend/assets/logo_symptom.jpg")
    img_summarizer = get_img_as_base64("frontend/assets/logo_summarizer.jpg")
except Exception as e:
    st.error(f"Error loading images: {e}")
    img_main = ""
    img_chatbot = ""
    img_risk = ""
    img_symptom = ""
    img_summarizer = ""

# Page config
st.set_page_config(
    page_title="GenMedX",
    page_icon=None,
    layout="wide"
)

# Custom CSS - Dark Professional Theme
st.markdown(f"""
<style>
    /* Global Styles */
    .main {{
        background-color: #0e1117;
    }}
    h1, h2, h3 {{
        color: #ecf0f1 !important;
        font-family: 'Inter', sans-serif;
    }}
    p, div {{
        color: #bdc3c7;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Header */
    .main-header {{
        text-align: center;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}
    .logo-container {{
        background-color: white;
        padding: 10px 20px;
        border-radius: 12px;
        display: inline-block;
        margin-bottom: 15px;
        box-shadow: 0 0 20px rgba(255,255,255,0.1);
    }}
    .sub-header {{
        font-size: 1.1rem;
        color: #95a5a6;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }}
    
    /* Cards */
    .card {{
        background-color: #1e2530;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #2c3e50;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        min-height: 320px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
    }}
    .card:hover {{
        transform: translateY(-5px);
        border-color: #3498db;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }}
    .card-img {{
        width: 80px;
        height: 80px;
        object-fit: contain;
        margin-bottom: 1.5rem;
        border-radius: 8px;
    }}
    .card-title {{
        font-size: 1.3rem;
        font-weight: 600;
        color: #ecf0f1;
        margin-bottom: 0.8rem;
    }}
    .card-desc {{
        color: #95a5a6;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    
    /* Buttons */
    .stButton>button {{
        width: 100%;
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        font-size: 1rem;
        margin-top: auto;
        transition: background-color 0.2s;
    }}
    .stButton>button:hover {{
        background-color: #2980b9;
    }}
</style>
""", unsafe_allow_html=True)

# Header with Logo
st.markdown(f"""
<div class="main-header">
    <div class="logo-container">
        <img src="data:image/jpeg;base64,{img_main}" style="height: 60px;">
    </div>
</div>
<p class="sub-header">Advanced Medical AI for Diagnosis & Risk Prediction</p>
""", unsafe_allow_html=True)

# System Status (Debug)
import config
with st.expander("🔌 System Status & API Keys"):
    st.write("Checking API Key Availability:")
    col_a, col_b = st.columns(2)
    
    def get_status(key_value):
        if key_value:
            return f"✅ Connected ({str(key_value)[:4]}...)"
        return "❌ Missing"

    with col_a:
        st.write(f"**Google API:** {get_status(config.GOOGLE_API_KEY)}")
        st.write(f"**Replicate API:** {get_status(config.REPLICATE_API_TOKEN)}")
    with col_b:
        st.write(f"**Qdrant URL:** {get_status(config.QDRANT_URL)}")
        st.write(f"**Qdrant Key:** {get_status(config.QDRANT_API_KEY)}")
    
    if not config.REPLICATE_API_TOKEN:
        st.error("Replicate Token is missing! Please check Streamlit Secrets.")
        
        # Debug: Show what keys ARE available
        if hasattr(st, "secrets"):
            st.write("---")
            st.write("**Debug Info: Available Secrets Keys**")
            found_keys = [k for k in st.secrets.keys() if not k.startswith("_")]
            if found_keys:
                for k in found_keys:
                    st.code(f"{k} = ...")
            else:
                st.warning("No secrets found in st.secrets!")
    else:
        st.info(f"Replicate Token detected. Length: {len(config.REPLICATE_API_TOKEN)}")

# Navigation Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="card">
        <div>
            <img src="data:image/jpeg;base64,{img_risk}" class="card-img">
            <div class="card-title">Risk Analysis</div>
            <div class="card-desc">
                Triage and risk prediction using patient vitals.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Analysis", use_container_width=True):
        st.switch_page("pages/1_Risk_Analysis.py")

with col2:
    st.markdown(f"""
    <div class="card">
        <div>
            <img src="data:image/jpeg;base64,{img_symptom}" class="card-img">
            <div class="card-title">Symptom Checker</div>
            <div class="card-desc">
                Preliminary diagnosis from symptoms.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Check Symptoms", use_container_width=True):
        st.switch_page("pages/2_Symptom_Checker.py")

with col3:
    st.markdown(f"""
    <div class="card">
        <div>
            <img src="data:image/jpeg;base64,{img_summarizer}" class="card-img">
            <div class="card-title">Stay Summarizer</div>
            <div class="card-desc">
                Automated patient discharge summaries.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Summarize Stay", use_container_width=True):
        st.switch_page("pages/3_Stay_Summarizer.py")

with col4:
    st.markdown(f"""
    <div class="card">
        <div>
            <img src="data:image/jpeg;base64,{img_chatbot}" class="card-img">
            <div class="card-title">Medical Chatbot</div>
            <div class="card-desc">
                AI-powered medical Q&A with RAG.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch Chatbot", use_container_width=True):
        st.switch_page("pages/4_Medical_Chatbot.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #57606f; padding: 2rem; font-size: 0.8rem;">
    Powered by GenMedX • BioMistral • Llama 3
</div>
""", unsafe_allow_html=True)
