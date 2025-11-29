import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Risk Analysis - AI Healthcare Agent",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #d32f2f 0%, #c62828 100%);
    }
    .risk-low {
        background: linear-gradient(135deg, #388e3c 0%, #2e7d32 100%);
    }
    .test-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.5rem;
        color: #333333; /* Fixed text color */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .reasoning-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bbdefb;
        color: #0d47a1;
    }
    .rag-source {
        font-size: 0.9rem;
        color: #555;
        background-color: #f5f5f5;
        padding: 0.8rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

st.markdown('<div class="main-header">Risk Analysis & Triage</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Vitals")
    
    with st.form("risk_form"):
        complaint = st.text_area("Chief Complaint", placeholder="e.g., Chest pain radiating to left arm, shortness of breath")
        
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
        
        submit = st.form_submit_button("Analyze Risk", use_container_width=True)

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
                    timeout=300  # Increased to 5 mins for Replicate cold boot
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Risk Score
                    risk_level = data['risk'].upper()
                    risk_class = "risk-high" if "HIGH" in risk_level else "risk-low"
                    
                    st.markdown(f"""
                    <div class="risk-card {risk_class}">
                        <h2>Risk Level: {risk_level}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Reasoning
                    st.markdown("### Clinical Reasoning")
                    st.markdown(f"""
                    <div class="reasoning-box">
                        {data['reasoning']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tests
                    st.markdown("### Recommended Tests")
                    for test in data['tests']:
                        st.markdown(f"""
                        <div class="test-card">
                            {test}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Similar Cases (RAG)
                    if data.get('similar_cases'):
                        st.markdown("### Relevant Medical Context (RAG)")
                        for case in data['similar_cases']:
                            st.markdown(f"""
                            <div class="rag-source">
                                <strong>Q: {case['question']}</strong><br>
                                <span style="color: #666;">{case['answer'][:200]}...</span>
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
