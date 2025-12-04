import streamlit as st
import sys
import os
import datetime

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import importlib
import modules.stay_summarizer.ml_model
import modules.stay_summarizer.agent
import backend.metrics

importlib.reload(modules.stay_summarizer.ml_model)
importlib.reload(modules.stay_summarizer.agent)
importlib.reload(backend.metrics)

from modules.stay_summarizer.agent import run_agent, run_manual_agent, get_available_patient_ids
from backend.metrics import metrics_tracker

st.set_page_config(
    page_title="Stay Summarizer",
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
    p, div, li, label {
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
    
    /* Summary Box */
    .summary-box {
        background-color: #1e2530;
        padding: 2rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin-top: 2rem;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2530;
        border-radius: 4px;
        padding: 10px 20px;
        color: #bdc3c7;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Stay Summarizer</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated patient discharge summaries using Fine-Tuned T5</p>', unsafe_allow_html=True)

# Tabs for Lookup vs Manual
tab1, tab2 = st.tabs(["Lookup by ID", "Manual Entry"])

# --- TAB 1: LOOKUP BY ID ---
with tab1:
    st.markdown("### Search Patient Database")
    
    available_ids = get_available_patient_ids()
    if available_ids:
        # Use a selectbox for ID to ensure valid input, but style it like a search
        selected_patient_id = st.selectbox(
            "Patient ID",
            options=available_ids,
            help="Select a patient from the available dataset."
        )
        
        # Main Content
        if selected_patient_id:
            st.markdown(f"### Patient ID: `{selected_patient_id}`")
            
            # Initialize session state for summary if not exists
            if "summary_result" not in st.session_state:
                st.session_state.summary_result = None
            
            if st.button("Generate Summary", type="primary", use_container_width=True):
                with st.spinner("Generating summary... (This may take a moment to download the model on first run)"):
                    try:
                        summary = run_agent(selected_patient_id)
                        st.session_state.summary_result = summary
                        st.session_state.feedback_submitted_lookup = False # Reset feedback
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            
            # Display Result from Session State
            if st.session_state.summary_result:
                st.markdown("### Generated Summary")
                st.markdown(f"""
                <div class="summary-box">
                    {st.session_state.summary_result.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                # Initialize feedback state
                if "feedback_submitted_lookup" not in st.session_state:
                    st.session_state.feedback_submitted_lookup = False
                
                st.markdown("### Rate this summary")
                
                if not st.session_state.feedback_submitted_lookup:
                    col_up, col_down = st.columns([1, 10])
                    with col_up:
                        if st.button("Helpful", key="like_lookup"):
                            metrics_tracker.log_feedback("Stay Summarizer", True)
                            st.session_state.feedback_submitted_lookup = True
                            st.rerun()
                    with col_down:
                        if st.button("Not Helpful", key="dislike_lookup"):
                            metrics_tracker.log_feedback("Stay Summarizer", False)
                            st.session_state.feedback_submitted_lookup = True
                            st.rerun()
                else:
                    st.info("Thanks for your feedback!")
    else:
        st.error("No patient data found. Please check the data source.")

# --- TAB 2: MANUAL ENTRY ---
with tab2:
    st.markdown("### Enter Patient Details")
    
    # Row 1: Date/Time & Transport
    col1, col2 = st.columns(2)
    with col1:
        visit_date = st.text_input("Visit Date/Time", placeholder="YYYY-MM-DD HH:MM:SS", value=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with col2:
        transport = st.text_input("Arrival Transport", placeholder="ambulance / walk-in")
        
    # Row 2: Chief Complaint
    complaint = st.text_input("Chief Complaint", placeholder="e.g., left hip pain after fall")
    
    # Row 3: Vitals (Temp, HR)
    col1, col2 = st.columns(2)
    with col1:
        temp = st.text_input("Temp (°F)", placeholder="98.6")
    with col2:
        hr = st.text_input("HR (/min)", placeholder="75")
        
    # Row 4: Vitals (RR, SpO2)
    col1, col2 = st.columns(2)
    with col1:
        rr = st.text_input("RR (/min)", placeholder="16")
    with col2:
        spo2 = st.text_input("SpO2 (%)", placeholder="97")
        
    # Row 5: Vitals (BP)
    col1, col2 = st.columns(2)
    with col1:
        sbp = st.text_input("SBP (mmHg)", placeholder="120")
    with col2:
        dbp = st.text_input("DBP (mmHg)", placeholder="80")
        
    # Row 6: Disposition & Diagnoses
    col1, col2 = st.columns(2)
    with col1:
        disposition = st.text_input("Disposition", placeholder="discharged home")
    with col2:
        diagnoses = st.text_input("Diagnoses", placeholder="e.g., Fracture; Fall")
        
    # Optional Question
    manual_question = st.text_input("Question (optional)", placeholder="Optional context", key="manual_q")
    
    # Initialize session state for manual summary
    if "manual_summary_result" not in st.session_state:
        st.session_state.manual_summary_result = None

    if st.button("Run Summary", type="primary", key="btn_manual"):
        with st.spinner("Generating summary..."):
            try:
                # Construct data dictionary
                stay_data = {
                    "intime": visit_date,
                    "arrival_transport": transport,
                    "chiefcomplaint": complaint,
                    "temperature": temp,
                    "heartrate": hr,
                    "resprate": rr,
                    "o2sat": spo2,
                    "sbp": sbp,
                    "dbp": dbp,
                    "disposition": disposition,
                    "diagnoses": diagnoses.split(";") if diagnoses else []
                }
                
                summary = run_manual_agent(stay_data)
                st.session_state.manual_summary_result = summary
                st.session_state.feedback_submitted_manual = False # Reset feedback
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display Manual Result
    if st.session_state.manual_summary_result:
        st.markdown("### Result")
        st.markdown(f"""
        <div class="summary-box">
            {st.session_state.manual_summary_result.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize feedback state
        if "feedback_submitted_manual" not in st.session_state:
            st.session_state.feedback_submitted_manual = False
        
        st.markdown("### Rate this summary")
        
        if not st.session_state.feedback_submitted_manual:
            col_up, col_down = st.columns([1, 10])
            with col_up:
                if st.button("Helpful", key="like_manual"):
                    metrics_tracker.log_feedback("Stay Summarizer", True)
                    st.session_state.feedback_submitted_manual = True
                    st.rerun()
            with col_down:
                if st.button("Not Helpful", key="dislike_manual"):
                    metrics_tracker.log_feedback("Stay Summarizer", False)
                    st.session_state.feedback_submitted_manual = True
                    st.rerun()
        else:
            st.info("Thanks for your feedback!")

