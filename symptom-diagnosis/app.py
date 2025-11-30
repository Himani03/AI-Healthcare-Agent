# ============================================================================
# MEDICAL DIAGNOSIS SYSTEM - HUGGING FACE SPACES
# Two-Phase Pipeline: BioMistral-7B Classification → GPT-4o Explanation
# ============================================================================

import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import PeftModel
import torch
from openai import OpenAI
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Sugandha-Chauhan/BioMistral-7B-SymptomDiagnosis"
BASE_MODEL = "BioMistral/BioMistral-7B"

# Get API keys from environment
HF_TOKEN = os.environ.get('HF_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

DIAGNOSIS_CLASSES = {
    0: "acute bronchitis",
    1: "anxiety",
    2: "conjunctivitis due to allergy",
    3: "eczema",
    4: "infectious gastroenteritis",
    5: "pneumonia",
    6: "psoriasis",
    7: "spondylosis",
    8: "sprain or strain",
    9: "strep throat"
}

# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading BioMistral-7B classification model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=10,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(model, MODEL_NAME, token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Model loaded on {device}")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
print("✓ OpenAI client initialized")

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def predict_diagnosis(symptoms_text):
    """Phase 1: Predict diagnosis using BioMistral-7B"""
    
    inputs = tokenizer(
        symptoms_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probabilities = torch.softmax(logits, dim=-1)
    confidence, predicted_class = torch.max(probabilities, dim=-1)
    
    diagnosis = DIAGNOSIS_CLASSES[predicted_class.item()]
    confidence_score = confidence.item()
    
    return diagnosis, confidence_score

def generate_explanation(diagnosis, symptoms):
    """Phase 2: Generate explanation using GPT-4o"""
    
    system_prompt = """You are a medical education assistant. Provide accurate, clear explanations of medical diagnoses based on symptoms. Your explanations must be evidence-based, accessible to general audiences, and include appropriate medical disclaimers."""
    
    user_prompt = f"""Generate a medical explanation for {diagnosis} based on these symptoms: {symptoms}

IMPORTANT: Do NOT include "Diagnosis:" or "Symptoms:" labels in your response. Start directly with the explanation.

Your response should cover:
1. Why these symptoms indicate this diagnosis (brief intro paragraph)
2. Pathophysiology: How the disease causes these symptoms
3. Management: Treatment options and self-care strategies
4. When to Seek Care: Warning signs requiring immediate medical attention
5. Disclaimer: Brief statement about consulting healthcare providers

Use clear language appropriate for a general audience. 300-400 words total.

Start your response directly with the explanation text."""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    
    return response.choices[0].message.content.strip()

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def diagnose_and_explain_interface(symptoms_text):
    """Main function for Gradio interface"""
    
    if not symptoms_text or symptoms_text.strip() == "":
        return "", "", "", ""
    
    try:
        # Phase 1: Classification
        diagnosis, confidence = predict_diagnosis(symptoms_text.strip())
        
        # Phase 2: Explanation
        explanation = generate_explanation(diagnosis, symptoms_text)
        
        # Format outputs
        diagnosis_result = f"**Diagnosis:** {diagnosis.title()}"
        confidence_result = f"**Confidence:** {confidence:.1%}"
        symptoms_display = f"**Symptoms:** {symptoms_text}"
        explanation_display = explanation  # Just the explanation, no extra formatting
        
        return diagnosis_result, confidence_result, symptoms_display, explanation_display
        
    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        return error_msg, "", "", ""

# ============================================================================
# BUILD INTERFACE
# ============================================================================

with gr.Blocks(title="GenMedX Medical Diagnosis System") as demo:
    
    # Header
    gr.Markdown("# Medical Diagnosis System")
    
    # Welcome message and instructions
    gr.Markdown("""
    Welcome to **GenMedX** medical diagnosis system. Enter your symptoms in the space below and make sure to follow the format.
    
    * ✓ **CORRECT INPUT:** nausea, vomiting, diarrhea, abdominal cramps
    * ✗ **INCORRECT INPUT:** I had nausea and vomiting last night...
    """)
    
    # Symptoms input
    symptoms_input = gr.Textbox(
        label="Enter Your Symptoms Here",
        placeholder="Example: nausea, vomiting, diarrhea, abdominal cramps",
        lines=3
    )
    
    # Diagnose button
    diagnose_btn = gr.Button("Diagnose", variant="primary", size="lg")
    
    # Results section
    gr.Markdown("## Results")
    diagnosis_output = gr.Markdown()
    confidence_output = gr.Markdown()
    
    # Medical Explanation section
    gr.Markdown("## Medical Explanation")
    symptoms_output = gr.Markdown()
    explanation_output = gr.Markdown()
    
    # Medical Disclaimer
    gr.Markdown("""
    ---
    **⚕️ MEDICAL DISCLAIMER:** This information is for educational purposes only and should not be used as a substitute for professional medical advice. Always consult with a healthcare provider for an accurate diagnosis and appropriate treatment options.
    """)
    
    # Connect button to function
    diagnose_btn.click(
        fn=diagnose_and_explain_interface,
        inputs=symptoms_input,
        outputs=[diagnosis_output, confidence_output, symptoms_output, explanation_output]
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch()