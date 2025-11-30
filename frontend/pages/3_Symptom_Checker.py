import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Medical Diagnosis System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS to match the dark theme and orange button design
st.markdown("""
<style>
    /* Force Dark Theme Backgrounds if not already set by Streamlit theme */
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    
    /* Header Styling */
    h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Input Instructions */
    .instruction-text {
        font-size: 1rem;
        color: #b0b0b0;
        margin-bottom: 1rem;
    }
    .correct-input {
        color: #4caf50; /* Green */
        font-weight: bold;
    }
    .incorrect-input {
        color: #f44336; /* Red */
        font-weight: bold;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #1e2329 !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 5px !important;
    }
    
    /* Orange Diagnose Button */
    .stButton > button {
        width: 100%;
        background-color: #ff6b00 !important; /* Orange */
        color: white !important;
        border: none !important;
        padding: 0.75rem 1rem !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        border-radius: 5px !important;
        margin-top: 1rem !important;
    }
    .stButton > button:hover {
        background-color: #e65100 !important; /* Darker Orange */
    }
    
    /* Results Section */
    .results-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Diagnosis Box */
    .diagnosis-box {
        background-color: #1e2329;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #ff6b00;
        margin-bottom: 1rem;
    }
    
    /* Disclaimer Footer */
    .disclaimer {
        font-size: 0.8rem;
        color: #888;
        margin-top: 3rem;
        border-top: 1px solid #333;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# Header
st.markdown("<h1>Medical Diagnosis System</h1>", unsafe_allow_html=True)

# Instructions
st.markdown("""
<div class="instruction-text">
    Welcome to GenMedX medical diagnosis system. Enter your symptoms in the space below and make sure to follow the format.<br>
    <ul>
        <li><span class="correct-input">✓ CORRECT INPUT:</span> nausea, vomiting, diarrhea, abdominal cramps</li>
        <li><span class="incorrect-input">✗ INCORRECT INPUT:</span> I had nausea and vomiting last night...</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Form
with st.form("diagnosis_form"):
    symptoms = st.text_area(
        "Enter Your Symptoms Here",
        placeholder="Example: nausea, vomiting, diarrhea, abdominal cramps",
        height=100,
        label_visibility="visible"
    )
    
    # Full width orange button
    submit = st.form_submit_button("Diagnose")

# Results Section
st.markdown('<div class="results-header">Results</div>', unsafe_allow_html=True)

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
                <div class="diagnosis-box">
                    <h2 style="margin:0; color: #ff6b00;">{diagnosis}</h2>
                    <p style="margin:0; color: #aaa;">Confidence: {confidence}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display Explanation
                st.markdown('<div class="results-header">Medical Explanation</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color: #1e2329; padding: 1.5rem; border-radius: 8px; color: #ddd; line-height: 1.6;">
                    {explanation}
                </div>
                """, unsafe_allow_html=True)
                
            else:
                error_detail = response.json().get('detail', response.text)
                if "PAUSED" in str(error_detail) or "timed out" in str(error_detail):
                     st.warning("⏳ The AI Model is waking up (Cold Boot). Please wait 1-2 minutes and try again.")
                else:
                    st.error(f"Error: {error_detail}")
                
        except Exception as e:
            st.error(f"Connection Error: {e}")

elif submit and not symptoms:
    st.warning("Please enter symptoms to proceed.")

# About the Model
with st.expander("ℹ️ How does this work? (Model & Data Source)"):
    st.markdown("""
    **Model Architecture:**
    *   **Core Model:** `BioMistral-7B` (Open-source medical LLM).
    *   **Fine-Tuning:** Specifically trained on a dataset of **10 common medical conditions** and their symptom patterns.
    *   **Explanation:** Uses `GPT-4o` to generate patient-friendly explanations based on the model's classification.
    
    **Data Source:**
    *   Unlike the Chatbot (which looks up a database), this module uses **Internal Model Knowledge** learned during training.
    *   It does *not* search the internet or a vector database.
    """)

# Footer
st.markdown("""
<div class="disclaimer">
    ⚕️ <strong>MEDICAL DISCLAIMER:</strong> This information is for educational purposes only and should not be used as a substitute for professional medical advice. Always consult with a healthcare provider for an accurate diagnosis and appropriate treatment options.
</div>
""", unsafe_allow_html=True)
