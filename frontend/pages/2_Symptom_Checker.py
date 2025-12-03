import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Symptom Checker",
    page_icon=None,
    layout="wide"
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
    p, div, label {
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
    
    /* Input Styling */
    .stTextArea>div>div>textarea {
        background-color: #1e2530;
        color: #ecf0f1;
        border: 1px solid #2c3e50;
        border-radius: 6px;
    }
    
    /* Content Boxes */
    .content-box {
        background-color: #1e2530;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #2c3e50;
        margin-bottom: 1rem;
    }
    .diagnosis-title {
        color: #3498db;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        color: #95a5a6;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .section-title {
        color: #ecf0f1;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid #2c3e50;
        padding-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
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

st.markdown('<div class="main-header">Symptom Checker</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Preliminary diagnosis and medical explanation based on symptoms</p>', unsafe_allow_html=True)

# Form
with st.form("diagnosis_form"):
    symptoms = st.text_area(
        "Describe Symptoms",
        placeholder="e.g., nausea, vomiting, diarrhea, abdominal cramps...",
        height=150
    )
    
    submit = st.form_submit_button("Analyze Symptoms")

# Results Section
if submit and symptoms:
    with st.spinner("Analyzing symptoms..."):
        try:
            response = requests.post(
                f"{API_URL}/symptom_predict",
                json={"symptoms": symptoms},
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                
                diagnosis = data.get('diagnosis', 'Unknown')
                confidence = data.get('confidence', '0%')
                explanation = data.get('explanation', 'No explanation provided.')
                
                # Display Diagnosis
                st.markdown(f"""
                <div class="content-box" style="text-align: center; border-left: 5px solid #3498db;">
                    <div class="diagnosis-title">{diagnosis}</div>
                    <p class="confidence-text">Confidence: {confidence}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display Explanation
                st.markdown(f"""
                <div class="content-box">
                    <div class="section-title">Medical Explanation</div>
                    <div style="line-height: 1.6;">{explanation}</div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                error_detail = response.json().get('detail', response.text)
                if "PAUSED" in str(error_detail) or "timed out" in str(error_detail):
                     st.warning("The AI Model is waking up. Please wait 1-2 minutes and try again.")
                else:
                    st.error(f"Error: {error_detail}")
                
        except Exception as e:
            st.error(f"Connection Error: {e}")

elif submit and not symptoms:
    st.warning("Please enter symptoms to proceed.")

# Footer
st.markdown("""
<div class="disclaimer">
    <strong>Medical Disclaimer:</strong> This AI assistant is for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)
