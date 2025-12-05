import streamlit as st
import pandas as pd
import plotly.express as px
import time
import sys
import os

# Add root directory to path so we can import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.metrics import metrics_tracker

st.set_page_config(
    page_title="Dashboard",
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
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #1e2530;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2c3e50;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        color: #95a5a6;
    }
    div[data-testid="stMetricValue"] {
        color: #ecf0f1;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
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

st.markdown('<div class="main-header">Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time system performance and architecture overview</p>', unsafe_allow_html=True)

# Auto-refresh mechanism
if st.button("Refresh Data"):
    st.rerun()

# Load Data
df = metrics_tracker.get_metrics_dataframe()

if df.empty:
    st.info("No performance data available yet. Run some predictions in the Risk Analysis or Symptom Checker modules to see metrics here!")
else:
    # --- TOP LEVEL METRICS ---
    # Filter for inference events for operational metrics
    if 'type' in df.columns:
        inference_df = df[df['type'] != 'feedback']
    else:
        inference_df = df
        
    total_requests = len(inference_df)
    avg_latency = inference_df['duration'].mean() if not inference_df.empty else 0
    error_count = inference_df[inference_df['success'] == False].shape[0]
    error_rate = (error_count / total_requests) * 100 if total_requests > 0 else 0
    
    # Calculate Satisfaction
    satisfaction_score = "No Data"
    if 'type' in df.columns and 'feedback' in df.columns:
        feedback_df = df[df['type'] == 'feedback']
        if not feedback_df.empty:
            positive_count = feedback_df[feedback_df['feedback'] == 'positive'].shape[0]
            total_feedback = len(feedback_df)
            score = (positive_count / total_feedback) * 100
            satisfaction_score = f"{score:.0f}%"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", total_requests)
    with col2:
        st.metric("Avg Inference Time", f"{avg_latency:.2f}s")
    with col3:
        st.metric("Error Rate", f"{error_rate:.1f}%")
    with col4:
        st.metric("User Satisfaction", satisfaction_score)

    st.markdown("---")

    # --- CHARTS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Inference Speed")
        # Line chart of duration over time
        fig_latency = px.line(
            df, 
            x='timestamp', 
            y='duration', 
            color='module',
            markers=True,
            title="Response Time per Request (Seconds)",
            labels={'duration': 'Seconds', 'timestamp': 'Time'},
            template="plotly_dark"
        )
        fig_latency.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_latency, use_container_width=True)

    with col_right:
        st.subheader("System Reliability")
        # Pie chart of Success vs Error
        status_counts = df['success'].value_counts().rename({True: 'Success', False: 'Error'})
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Success vs Error Rate",
            color=status_counts.index,
            color_discrete_map={'Success': '#27ae60', 'Error': '#c0392b'},
            template="plotly_dark"
        )
        fig_status.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_status, use_container_width=True)

    # --- MODULE BREAKDOWN ---
    st.subheader("Module Traffic")
    module_counts = df['module'].value_counts().reset_index()
    module_counts.columns = ['Module', 'Requests']
    
    fig_traffic = px.bar(
        module_counts, 
        x='Module', 
        y='Requests',
        color='Module',
        title="Total Requests by Module",
        template="plotly_dark"
    )
    fig_traffic.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_traffic, use_container_width=True)

    # --- RECENT LOGS ---
    with st.expander("View Raw Logs"):
        st.dataframe(df.sort_values('timestamp', ascending=False), use_container_width=True)

st.markdown("---")

# ==========================================
# GENERAL SYSTEM PIPELINES (Preserved Context)
# ==========================================
st.header("System Architecture Context")

# Architecture Diagram
st.markdown("### High-Level Design")
st.markdown("""
```mermaid
graph TD
    User([User]) <--> Frontend[Streamlit Frontend]
    Frontend <-->|REST API (JSON)| Backend[FastAPI Backend]
    
    subgraph "Core Application"
        Backend --> Router{API Router}
        Router -->|/chat| ChatEngine[Chatbot Engine]
        Router -->|/risk_predict| RiskEngine[Risk Analysis Engine]
        Router -->|/symptom_predict| SymptomEngine[Symptom Diagnosis Engine]
        Router -->|/summarize| StayEngine[Stay Summarizer Engine]
    end
    
    subgraph "Data Layer"
        ChatEngine <-->|Retrieve| Qdrant[Qdrant Vector DB]
        RiskEngine <-->|Retrieve| Qdrant
    end
    
    subgraph "Inference Layer (Cloud)"
        ChatEngine -->|Generate| Replicate[Replicate (Llama 3)]
        RiskEngine -->|Predict| HF_Space1[HF Space: BioMistral Adapter]
        SymptomEngine -->|Classify| HF_Space2[HF Space: Symptom Model]
        SymptomEngine -->|Explain| OpenAI[OpenAI (GPT-4o)]
        StayEngine -->|Summarize| HF_Space3[HF Space: T5-Small]
    end
```
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Unified Data Pipeline", "Chatbot Engine", "Risk Analysis Engine", "Symptom Diagnosis Engine", "Stay Summarizer"])

with tab1:
    st.markdown("""
    ### Multi-Stream Data Processing
    The system ingests data from distinct sources to power its modules:
    
    **1. Medical Knowledge Base (for Chatbot)**
    *   **Sources**: MedQA-USMLE (11k pairs), PubMedQA.
    *   **Process**: MCQ to Q&A conversion → Embedding → Qdrant (`medical_knowledge_base`).
    
    **2. Triage Historical Data (for Risk Analysis)**
    *   **Sources**: MIMIC-IV Emergency Admissions (Synthetic subset).
    *   **Process**: Vitals/Complaint extraction → Embedding → Qdrant (`triage_cases`).

    **3. Symptom-Disease Patterns (for Symptom Diagnosis)**
    *   **Sources**: 10 Common Disease Datasets (Symptom-Label pairs).
    *   **Process**: Fine-tuning BioMistral-7B (LoRA adapters).
    *   **Storage**: Model Weights (Hugging Face Hub).

    **4. Clinical Summarization Data (for Stay Summarizer)**
    *   **Sources**: MIMIC-IV Discharge Summaries.
    *   **Process**: Fine-tuning T5-Small (Seq2Seq).
    *   **Storage**: Model Weights (Hugging Face Hub).
    
    **Infrastructure**
    *   **Vector DB**: Qdrant Cloud (High-performance similarity search).
    *   **Model Hub**: Hugging Face (Storing fine-tuned adapters).
    *   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-d).
    """)

with tab2:
    st.markdown("""
    ### Medical Q&A Pipeline
    A specialized RAG system powered by a fine-tuned medical LLM.
    
    1.  **Retrieval**: Fetches top 5 relevant medical Q&A pairs from the Qdrant knowledge base.
    2.  **Context Window**: Injects retrieved knowledge into the system prompt.
    3.  **Inference**: 
        *   **Meditron**: A state-of-the-art open-source medical LLM.
        *   **Fine-Tuning**: The model has been further fine-tuned on the **MedQA-USMLE** dataset to enhance its medical reasoning and accuracy.
    4.  **Output**: Generates answer with citations to the source documents.
    """)

with tab3:
    st.markdown("""
    ### Intelligent Risk Analysis Pipeline
    A specialized "Verify First" architecture to prevent hallucinations.
    
    1.  **Local RAG**: Retrieves similar historical cases based on "Chief Complaint".
    2.  **Prompt Injection**: Constructs a strict "Chain of Thought" prompt locally.
        *   *Instruction*: "First, verify patient vitals against normal ranges."
    3.  **Cloud Inference**: Injects the full prompt into the **BioMistral Adapter** (Hugging Face Space).
    4.  **Parsing**: Extracts Risk Level (High/Low) and structured reasoning from the model's output.
    """)

with tab4:
    st.markdown("""
    ### Hybrid Symptom Diagnosis Pipeline
    A two-stage AI pipeline combining specialized classification with generative explanation.
    
    1.  **Symptom Input**: User describes symptoms (e.g., "nausea, vomiting").
    2.  **Classification (BioMistral-7B)**: 
        *   Fine-tuned on 10 specific disease classes.
        *   Predicts the most likely condition (e.g., "Infectious Gastroenteritis").
    3.  **Explanation (GPT-4o)**:
        *   Takes the predicted diagnosis and original symptoms.
        *   Generates a patient-friendly explanation, pathophysiology, and care advice.
    4.  **Output**: Structured Diagnosis Card + Medical Explanation.
    """)

with tab5:
    st.markdown("""
    ### Automated Stay Summarizer Pipeline
    A specialized Seq2Seq pipeline for generating concise hospital stay summaries.
    
    1.  **Input Methods**: 
        *   **CSV Lookup**: Retrieves structured patient history (Vitals, Labs, Diagnosis) from a local dataset.
        *   **Manual Entry**: Allows real-time input of patient parameters via a dynamic form.
    2.  **Serialization**: 
        *   Converts structured data into a linear text format optimized for the model.
        *   *Format*: `DATE: ... | COMPLAINT: ... | VITALS: ... | DIAGNOSES: ...`
    3.  **Inference (Fine-Tuned T5)**:
        *   **Model**: `kushbindal/genmedx-t5-small` (Hugging Face).
        *   **Fine-Tuning**: Trained on MIMIC-IV discharge summaries to learn medical summarization patterns.
        *   **Logic**: Uses a "Diagnosis Patch" to ensure critical diagnoses are never omitted from the summary.
    4.  **Output**: A professional, paragraph-style summary ready for medical handoffs.
    """)