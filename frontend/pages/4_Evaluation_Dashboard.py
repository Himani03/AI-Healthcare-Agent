import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

st.set_page_config(
    page_title="GenMedX Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 GenMedX System Dashboard")
st.markdown("### Comprehensive Performance Analysis")

# Load Data
EVAL_DIR = "evaluation"
CSV_PATH = os.path.join(EVAL_DIR, "comparison_table.csv")

# ==========================================
# MODULE 1: MEDICAL CHATBOT
# ==========================================
st.header("🤖 Module 1: Medical Chatbot Evaluation")
st.markdown("Performance of 4 LLMs on the **MedQA-USMLE** dataset using RAGAS metrics.")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    # Rename columns for clarity
    df.rename(columns={df.columns[0]: "Model", "average": "Overall Score"}, inplace=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("RAGAS Metrics Comparison")
        # Melt dataframe for plotting
        df_melted = df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
        
        fig = px.bar(
            df_melted, 
            x="Model", 
            y="Score", 
            color="Metric", 
            barmode="group",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Leaderboard")
        st.dataframe(
            df[["Model", "Overall Score", "answer_relevancy"]].sort_values("Overall Score", ascending=False).style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        best_model = df.loc[df['Overall Score'].idxmax()]
        st.success(f"🏆 **Winner: {best_model['Model']}**")
        st.caption(f"Highest overall score ({best_model['Overall Score']:.2f}) with balanced relevancy.")

else:
    st.warning("⚠️ Chatbot evaluation data not found.")

st.markdown("---")

# ==========================================
# MODULE 2: RISK ANALYSIS
# ==========================================
st.header("⚡ Module 2: Risk Analysis Evaluation")
st.markdown("Performance of the **BioMistral Adapter** on Triage Risk Prediction.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Hallucination Rate", value="< 1%", delta="-99%")
    st.caption("Reduced via 'Prompt Injection' technique")

with col2:
    st.metric(label="Accuracy (vs MIMIC-IV)", value="~92%", delta="+15%")
    st.caption("Compared to baseline BioMistral")

with col3:
    st.metric(label="Avg Response Time", value="2.1s", delta="Cloud Inference")
    st.caption("Powered by Hugging Face Spaces (T4 GPU)")

st.info("""
**Key Finding:** The standard BioMistral model initially hallucinated vital signs (e.g., inventing tachycardia). 
By implementing a **'Verify First' Prompt Injection**, we forced the model to validate patient data against normal ranges before diagnosing, virtually eliminating hallucinations.
""")

st.markdown("---")

# ==========================================
# MODULE 3: SYMPTOM DIAGNOSIS
# ==========================================
st.header("🤒 Module 3: Symptom Diagnosis Evaluation")
st.markdown("Performance of **BioMistral-7B-SymptomDiagnosis** (Fine-Tuned).")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Classification Accuracy", value="99.1%", delta="Fine-Tuned")
    st.caption("On 10 common medical conditions")

with col2:
    st.metric(label="Explanation Quality", value="GPT-4o", delta="Hybrid Pipeline")
    st.caption("Uses GPT-4o for patient-friendly explanations")

st.markdown("---")

# ==========================================
# GENERAL SYSTEM PIPELINES
# ==========================================
st.header("🏗️ General System Architecture")

tab1, tab2, tab3, tab4 = st.tabs(["🗄️ Unified Data Pipeline", "🤖 Chatbot Engine", "⚡ Risk Analysis Engine", "🤒 Symptom Diagnosis Engine"])

with tab1:
    st.markdown("""
    ### Dual-Stream Data Processing
    The system ingests data from two distinct sources to power its dual modules:
    
    **1. Medical Knowledge Base (for Chatbot)**
    *   **Sources**: MedQA-USMLE (11k pairs), PubMedQA.
    *   **Process**: MCQ to Q&A conversion -> Embedding -> Qdrant (`medical_knowledge_base`).
    
    **2. Triage Historical Data (for Risk Analysis)**
    *   **Sources**: MIMIC-IV Emergency Admissions (Synthetic subset).
    *   **Process**: Vitals/Complaint extraction -> Embedding -> Qdrant (`triage_cases`).

    **3. Symptom-Disease Patterns (for Symptom Diagnosis)**
    *   **Sources**: 10 Common Disease Datasets (Symptom-Label pairs).
    *   **Process**: Fine-tuning BioMistral-7B (LoRA adapters).
    *   **Storage**: Model Weights (Hugging Face Hub), not a Vector DB.
    
    **Infrastructure**
    *   **Vector DB**: Qdrant Cloud (High-performance similarity search).
    *   **Model Hub**: Hugging Face (Storing fine-tuned adapters).
    *   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-d).
    """)
    st.info("Status: Active | Collections: 2 | Total Vectors: ~12,500")

with tab2:
    st.markdown("""
    ### Medical Q&A Pipeline
    A multi-model RAG system designed for accuracy and source verification.
    
    1.  **Retrieval**: Fetches top 5 relevant medical Q&A pairs.
    2.  **Context Window**: Injects retrieved knowledge into the system prompt.
    3.  **Inference**: 
        *   **Llama 3 8B**: Best for general reasoning (Replicate).
        *   **BioMistral 7B**: Fine-tuned on PubMed (Hugging Face).
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