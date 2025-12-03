import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Risk Analysis",
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
    
    /* Input Cards */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1e2530;
        color: #ecf0f1;
        border: 1px solid #2c3e50;
        border-radius: 6px;
    }
    
    /* Risk Badges */
    .risk-badge {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #c0392b 0%, #a93226 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(192, 57, 43, 0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #d35400 0%, #ba4a00 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(211, 84, 0, 0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #27ae60 0%, #219150 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
    }
    .risk-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .risk-prob {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Content Boxes */
    .content-box {
        background-color: #1e2530;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #2c3e50;
        margin-bottom: 1rem;
    }
    .section-title {
        color: #3498db;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

st.markdown('<div class="main-header">Risk Analysis</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered triage assessment based on patient vitals</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Patient Vitals")
    
    with st.form("risk_form"):
        complaint = st.text_area("Chief Complaint", placeholder="e.g., Chest pain radiating to left arm...")
        
        c1, c2 = st.columns(2)
        with c1:
            temp = st.number_input("Temperature (°F)", 90.0, 110.0, 98.6)
            hr = st.number_input("Heart Rate (bpm)", 30, 250, 80)
            rr = st.number_input("Resp Rate (bpm)", 5, 60, 16)
        
        with c2:
            o2 = st.number_input("O2 Saturation (%)", 50, 100, 98)
            sbp = st.number_input("Systolic BP (mmHg)", 50, 250, 120)
            dbp = st.number_input("Diastolic BP (mmHg)", 30, 150, 80)
            
        pain = st.slider("Pain Level (0-10)", 0, 10, 0)
        
        submit = st.form_submit_button("Analyze Risk")

with col2:
    if submit and complaint:
        with st.spinner("Analyzing clinical data..."):
            try:
                response = requests.post(
                    f"{API_URL}/risk_predict",
                    json={
                        "complaint": complaint,
                        "vitals": {
                            "temperature": temp,
                            "heartrate": hr,
                            "resprate": rr,
                            "o2sat": o2,
                            "sbp": sbp,
                            "dbp": dbp,
                            "pain": pain
                        }
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse Result
                    risk_level = result.get('risk', 'Unknown')
                    probability = result.get('probability', 'N/A')
                    reasoning = result.get('reasoning', 'No reasoning provided.')
                    tests = result.get('tests', [])
                    rag_html = result.get('rag_html', '')
                    
                    # Display Risk Badge
                    badge_class = "risk-low"
                    if "High" in risk_level:
                        badge_class = "risk-high"
                    elif "Medium" in risk_level:
                        badge_class = "risk-medium"
                        
                    st.markdown(f"""
                    <div class="risk-badge {badge_class}">
                        <div class="risk-title">{risk_level}</div>
                        <div class="risk-prob">Probability: {probability}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display Reasoning
                    st.markdown(f"""
                    <div class="content-box">
                        <div class="section-title">Clinical Reasoning</div>
                        {reasoning}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display Tests
                    if tests:
                        tests_html = "".join([f"<li>{test}</li>" for test in tests])
                        st.markdown(f"""
                        <div class="content-box">
                            <div class="section-title">Recommended Tests</div>
                            <ul style="margin-bottom: 0;">{tests_html}</ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Display RAG Context
                    st.markdown(f"""
                    <div class="content-box">
                        <div class="section-title">Similar Historical Cases</div>
                        {rag_html}
                    </div>
                    """, unsafe_allow_html=True)
                            
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")
    
    elif submit and not complaint:
        st.warning("Please enter a chief complaint.")
    
    else:
        st.info("Enter patient data to generate risk analysis.")
