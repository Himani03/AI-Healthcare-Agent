"""
FastAPI Backend for AI Healthcare Agent
Main API server with chat endpoint
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.shared.models import ModelManager
from rag.retriever import RAGRetriever
from backend.risk_predictor import get_risk_predictor
from backend.symptom_predictor import get_symptom_predictor
from backend.metrics import metrics_tracker

# Initialize FastAPI app
app = FastAPI(
    title="AI Healthcare Agent API",
    description="Medical Q&A chatbot with RAG and multiple LLM models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_manager = None
rag_retriever = None
risk_predictor = None
symptom_predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and RAG on startup"""
    global model_manager, rag_retriever, risk_predictor, symptom_predictor
    
    print("🚀 Starting AI Healthcare Agent API...")
    model_manager = ModelManager()
    rag_retriever = RAGRetriever()
    
    # Initialize Risk Predictor (lazy load or startup)
    try:
        risk_predictor = get_risk_predictor()
        print("✅ Risk Predictor ready!")
    except Exception as e:
        print(f"⚠️ Risk Predictor failed to load: {e}")

    # Initialize Symptom Predictor
    try:
        symptom_predictor = get_symptom_predictor()
        print("✅ Symptom Predictor ready!")
    except Exception as e:
        print(f"⚠️ Symptom Predictor failed to load: {e}")
        
    print("✅ API ready!")

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    model: str = "gemini"  # gemini, llama, biomistral, meditron
    use_rag: bool = True

class ChatResponse(BaseModel):
    answer: str
    model: str
    model_name: str
    latency_ms: float
    citations: list
    used_rag: bool
    error: bool = False

class RiskRequest(BaseModel):
    complaint: str
    vitals: dict

class RiskResponse(BaseModel):
    risk: str
    reasoning: str
    tests: list
    similar_cases: list
    error: bool = False

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Healthcare Agent API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "models": "/models",
            "health": "/health",
            "risk": "/risk_predict"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": model_manager is not None,
        "rag_loaded": rag_retriever is not None,
        "risk_loaded": risk_predictor is not None
    }

@app.get("/models")
async def get_models():
    """Get available models"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    models = []
    for model_id in model_manager.get_available_models():
        info = model_manager.get_model_info(model_id)
        models.append({
            "id": model_id,
            "name": info.get('name', ''),
            "type": info.get('type', ''),
            "description": info.get('description', '')
        })
    
    return {"models": models}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    """
    if not model_manager or not rag_retriever:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = time.time()
    
    # Retrieve context if RAG enabled
    context = ""
    citations = []
    if request.use_rag:
        context, results = rag_retriever.retrieve(request.question)
        citations = rag_retriever.get_citations(results)
    
    # Generate answer
    result = model_manager.generate(
        model_name=request.model,
        question=request.question,
        context=context,
        use_rag=request.use_rag
    )
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    duration_sec = time.time() - start_time
    
    # Add medical disclaimer
    answer = result['answer']
    # Disclaimer removed (handled by frontend footer)
    
    # Log metrics
    success = not result.get('error', False)
    error_msg = result.get('error_msg') if not success else None
    metrics_tracker.log_inference("Medical Chatbot", duration_sec, success, error_msg)

    return ChatResponse(
        answer=answer,
        model=result['model'],
        model_name=result.get('model_name', result['model']),
        latency_ms=latency_ms,
        citations=citations,
        used_rag=request.use_rag,
        error=result.get('error', False)
    )

@app.post("/risk_predict")
async def predict_risk(request: RiskRequest):
    """
    Risk prediction endpoint
    """
    if not risk_predictor:
        raise HTTPException(status_code=503, detail="Risk Predictor not available")
    try:
        result = risk_predictor.predict(request.complaint, request.vitals)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SymptomRequest(BaseModel):
    symptoms: str

@app.post("/symptom_predict")
async def predict_symptoms(request: SymptomRequest):
    """
    Symptom prediction endpoint
    """
    try:
        # Note: This will fail if the HF Space is paused
        result = symptom_predictor.predict(request.symptoms)
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn backend.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
