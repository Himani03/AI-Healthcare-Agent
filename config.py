"""
Configuration file for AI Healthcare Agent
Add your API keys here
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# API KEYS - ADD YOUR KEYS HERE
# ============================================

# Google AI (for Gemini 1.5 Flash)
# Get from: https://makersuite.google.com/app/apikey
import streamlit as st

# Helper to get secret/env var
def get_secret(key, default=""):
    # 1. Try Streamlit Secrets
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    # 2. Try Environment Variable
    return os.getenv(key, default)

# ============================================
# API KEYS - ADD YOUR KEYS HERE
# ============================================

# Google AI (for Gemini 1.5 Flash)
# Get from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")

# Replicate (for Llama, BioMistral, Meditron)
# Get from: https://replicate.com/account/api-tokens
REPLICATE_API_TOKEN = get_secret("REPLICATE_API_TOKEN")

# Qdrant Cloud (Vector Database)
# Get from: https://cloud.qdrant.io
QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")

# OpenAI (for RAGAS evaluation)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

# ============================================
# MODEL CONFIGURATIONS
# ============================================

MODELS = {
    "gemini": {
        "name": "Gemini 1.5 Flash",
        "type": "general",
        "provider": "google",
        "model_id": "gemini-flash-latest",
        "description": "General-purpose LLM, strong reasoning"
    },
    "llama": {
        "name": "Llama 3 8B",
        "type": "open_source",
        "provider": "replicate",
        "model_id": "meta/meta-llama-3-8b-instruct",
        "version": "5a6809ca6288247d06daf6365557e5e429063f32a21146b2a807c682652136b8",
        "description": "Powerful open model by Meta"
    },
    "biomistral": {
        "name": "Mistral 7B",
        "type": "medical",
        "provider": "replicate",
        "model_id": "nateraw/mistral-7b-openorca",
        "version": "7afe21847d582f7811327c903433e29334c31fe861a7cf23c62882b181bacb88",
        "description": "Efficient 7B model (OpenOrca)"
    },
    "meditron": {
        "name": "Llama 3 70B",
        "type": "medical",
        "provider": "replicate",
        "model_id": "meta/meta-llama-3-70b-instruct",
        "version": "fbfb20b472b2f3bdd101412a9f70a0ed4fc0ced78a77ff00970ee7a2383c575d",
        "description": "High-performance large model"
    }
}

# ============================================
# RAG CONFIGURATION
# ============================================

RAG_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",  # Free, 384 dimensions
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5,  # Number of documents to retrieve
    "collection_name": "medical_qa",
    "triage_collection": "triage_cases"
}

# ============================================
# GENERATION PARAMETERS
# ============================================

GENERATION_PARAMS = {
    "temperature": 0.7,  # Balance between creativity and consistency
    "max_tokens": 512,
    "top_p": 0.9
}

# ============================================
# PROMPTS (Prompt Engineering for RAG)
# ============================================

SYSTEM_PROMPT = """You are a helpful medical AI assistant. Your role is to provide accurate, clear medical information based on the provided context.

IMPORTANT GUIDELINES:
1. Use ONLY the information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer that"
3. Provide clear, patient-friendly explanations
4. Include relevant citations from the context

Remember: This is for educational purposes only and not a substitute for professional medical advice."""

RAG_PROMPT_TEMPLATE = """Context from medical knowledge base:
{context}

Question: {question}

Instructions:
- Answer the question clearly and helpfully based on the context.
- Provide a direct answer first, then explain the reasoning or details.
- Include relevant medical context (e.g., potential causes, next steps) if available in the source.
- Avoid generic filler, but ensure the answer is complete and informative.
- Cite sources when possible.

Answer:"""

# ============================================
# EVALUATION CONFIGURATION
# ============================================

EVAL_CONFIG = {
    "test_set_size": 100,  # Number of questions for evaluation
    "ragas_metrics": [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness"
    ]
}

# ============================================
# VALIDATION
# ============================================

def validate_config():
    """Validate that all required API keys are set"""
    missing_keys = []
    
    if not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    if not REPLICATE_API_TOKEN:
        missing_keys.append("REPLICATE_API_TOKEN")
    if not QDRANT_URL:
        missing_keys.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing_keys.append("QDRANT_API_KEY")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY (needed for RAGAS evaluation)")
    
    if missing_keys:
        print("⚠️  WARNING: Missing API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease add them to your .env file")
        return False
    
    print("✅ All API keys configured!")
    return True

if __name__ == "__main__":
    validate_config()
