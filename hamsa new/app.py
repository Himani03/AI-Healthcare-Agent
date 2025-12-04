import gradio as gr
import torch
import pandas as pd
import numpy as np
import re
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress logs
if sys.platform == "linux":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_MODEL_ID = "BioMistral/BioMistral-7B"
ADAPTER_ID = "hamsaram/GenMedX-Adapter" 
DATA_FILE = "train_preprocessed.csv" 

print("Initializing GenMedX...")

# --- A. Load LLM ---
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading Adapter: {ADAPTER_ID}...")
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_ID)
        print("Custom Adapter Loaded!")
    except Exception as e:
        print(f"Adapter Error: {e}")

except Exception as e:
    print(f"Critical Model Error: {e}")
    raise e

# --- B. Load RAG System ---
print("Building RAG Search Engine...")
df_rag = None
df_raw_vitals = None
rag_embeddings = None
text_col = None 
embedder = None  # Define embedder globally

try:
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
    else:
        print(f"Reading {DATA_FILE}...")
        df_rag = pd.read_csv(DATA_FILE, engine='python', on_bad_lines='skip')
        
        # Load raw vitals from triage.csv
        RAW_DATA_FILE = "triage.csv"
        if os.path.exists(RAW_DATA_FILE):
            print(f"Loading raw vitals from {RAW_DATA_FILE}...")
            df_raw_vitals = pd.read_csv(RAW_DATA_FILE, engine='python', on_bad_lines='skip')
            # Keep only necessary columns
            vital_cols = ['subject_id', 'stay_id', 'temperature', 'heartrate', 'resprate', 
                         'o2sat', 'sbp', 'dbp', 'pain']
            df_raw_vitals = df_raw_vitals[vital_cols].drop_duplicates(subset=['subject_id', 'stay_id'])
            print(f"Loaded {len(df_raw_vitals)} raw vital records")
        else:
            print(f"Warning: {RAW_DATA_FILE} not found, will use preprocessed values")
        
        target_cols = ['chiefcomplaint_clean', 'chiefcomplaint', 'text', 'input']
        text_col = next((col for col in target_cols if col in df_rag.columns), None)
        
        if text_col:
            print(f"Indexing embeddings...")
            # Deduplicate strictly on the text column to avoid 3 identical results
            df_rag = df_rag.drop_duplicates(subset=[text_col])
            
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            texts = df_rag[text_col].astype(str).fillna("").tolist()
            rag_embeddings = embedder.encode(texts, show_progress_bar=True)
            print("RAG Database Ready")
        else:
            print(f"RAG Error: Text columns not found. Available: {list(df_rag.columns)}")
            df_rag = None

except Exception as e:
    print(f"RAG Crash: {str(e)}")

# ==========================================
# 2. LOGIC FUNCTIONS
# ==========================================

def validate_vitals_and_add_context(temp, hr, rr, o2, sbp, dbp, pain):
    """Generate accurate vital signs context based on actual ranges"""
    context = []
    
    # Temperature: 97-99°F normal
    if temp < 97:
        context.append(f"Temperature {temp}°F is BELOW normal range (hypothermia risk)")
    elif temp > 99:
        context.append(f"Temperature {temp}°F is ABOVE normal range (fever)")
    else:
        context.append(f"Temperature {temp}°F is within normal range")
    
    # Heart Rate: 60-100 bpm normal
    if hr < 60:
        context.append(f"Heart rate {hr} bpm is BELOW normal range (bradycardia)")
    elif hr > 100:
        context.append(f"Heart rate {hr} bpm is ABOVE normal range (tachycardia)")
    else:
        context.append(f"Heart rate {hr} bpm is within normal range")
    
    # Respiratory Rate: 12-20 /min normal
    if rr < 12:
        context.append(f"Respiratory rate {rr}/min is BELOW normal range (bradypnea)")
    elif rr > 20:
        context.append(f"Respiratory rate {rr}/min is ABOVE normal range (tachypnea)")
    else:
        context.append(f"Respiratory rate {rr}/min is within normal range")
    
    # O2 Saturation: 95-100% normal
    if o2 < 95:
        context.append(f"Oxygen saturation {o2}% is BELOW normal range (hypoxia)")
    else:
        context.append(f"Oxygen saturation {o2}% is within normal range")
    
    # Systolic BP: 90-120 mmHg normal
    if sbp < 90:
        context.append(f"Systolic BP {sbp} mmHg is BELOW normal range (hypotension)")
    elif sbp > 120:
        context.append(f"Systolic BP {sbp} mmHg is ABOVE normal range (hypertension)")
    else:
        context.append(f"Systolic BP {sbp} mmHg is within normal range")
    
    # Diastolic BP: 60-80 mmHg normal
    if dbp < 60:
        context.append(f"Diastolic BP {dbp} mmHg is BELOW normal range")
    elif dbp > 80:
        context.append(f"Diastolic BP {dbp} mmHg is ABOVE normal range")
    else:
        context.append(f"Diastolic BP {dbp} mmHg is within normal range")
    
    # Pain: 0-3 mild, 4-7 moderate, 8-10 severe
    if pain == 0:
        context.append(f"Pain score {pain}/10 (no pain)")
    elif pain <= 3:
        context.append(f"Pain score {pain}/10 (mild pain)")
    elif pain <= 7:
        context.append(f"Pain score {pain}/10 (moderate pain)")
    else:
        context.append(f"Pain score {pain}/10 (severe pain)")
    
    return "; ".join(context)

def check_vitals_logic(temp, hr, rr, o2, sbp, dbp):
    alerts = []
    if temp > 100.4: alerts.append(f"FEVER ({temp}°F)")
    elif temp < 95.0: alerts.append(f"HYPOTHERMIA ({temp}°F)")
    
    if hr > 100: alerts.append(f"TACHYCARDIA (HR {hr})")
    elif hr < 60: alerts.append(f"BRADYCARDIA (HR {hr})")
    
    if rr > 20: alerts.append(f"TACHYPNEA (RR {rr})")
    elif rr < 12: alerts.append(f"BRADYPNEA (RR {rr})")
    
    if o2 < 95: alerts.append(f"HYPOXIA (O2 {o2}%)")
    
    if sbp > 140 or dbp > 90: alerts.append(f"HYPERTENSION ({sbp}/{dbp})")
    elif sbp < 90: alerts.append(f"HYPOTENSION ({sbp}/{dbp})")
    
    if not alerts: 
        return "Vitals within normal limits."
    else: 
        return "CRITICAL ABNORMALITIES: " + ", ".join(alerts)

def retrieve_similar_cases(query, k=3):
    if df_rag is None or embedder is None: 
        return "RAG Database not active."
    
    query_vec = embedder.encode([str(query)])
    similarities = cosine_similarity(query_vec, rag_embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    results = []
    for i, idx in enumerate(top_indices):
        row = df_rag.iloc[idx]
        score = similarities[idx]
        
        # --- FIX: Cap Score at 99% to avoid misleading 100% matches ---
        if score >= 0.99:
            score = 0.99
        
        subj_id = row.get('subject_id', 'N/A')
        stay_id = row.get('stay_id', 'N/A')
        outcome = row.get('risk_label', row.get('acuity', 'N/A'))
        complaint_txt = str(row.get(text_col, 'N/A'))[:100] + "..."
        
        # --- NEW: Get raw vitals from triage.csv using subject_id and stay_id ---
        if df_raw_vitals is not None and subj_id != 'N/A' and stay_id != 'N/A':
            # Find matching row in raw vitals
            match = df_raw_vitals[
                (df_raw_vitals['subject_id'] == subj_id) & 
                (df_raw_vitals['stay_id'] == stay_id)
            ]
            
            if not match.empty:
                raw_row = match.iloc[0]
                temp_val = raw_row.get('temperature', 'N/A')
                hr_val = raw_row.get('heartrate', 'N/A')
                rr_val = raw_row.get('resprate', 'N/A')
                o2_val = raw_row.get('o2sat', 'N/A')
                sbp_val = raw_row.get('sbp', 'N/A')
                dbp_val = raw_row.get('dbp', 'N/A')
                pain_val = raw_row.get('pain', 'N/A')
            else:
                # Fallback if no match found
                temp_val = hr_val = rr_val = o2_val = sbp_val = dbp_val = pain_val = 'N/A'
        else:
            # Fallback to preprocessed data (will show negative values but better than nothing)
            temp_val = row.get('temperature', row.get('temperature_raw', 'N/A'))
            hr_val = row.get('heartrate', row.get('heartrate_raw', 'N/A'))
            rr_val = row.get('resprate', row.get('resprate_raw', 'N/A'))
            o2_val = row.get('o2sat', row.get('o2sat_raw', 'N/A'))
            sbp_val = row.get('sbp', row.get('sbp_raw', 'N/A'))
            dbp_val = row.get('dbp', row.get('dbp_raw', 'N/A'))
            pain_val = row.get('pain', row.get('pain_raw', 'N/A'))
        
        # Format vitals nicely - handle N/A and round numbers
        def format_vital(val):
            if val == 'N/A' or pd.isna(val):
                return 'N/A'
            try:
                return f"{float(val):.1f}"
            except:
                return str(val)
        
        vitals_str = (f"T={format_vital(temp_val)}°F, HR={format_vital(hr_val)}, "
                     f"RR={format_vital(rr_val)}, O2={format_vital(o2_val)}%, "
                     f"BP={format_vital(sbp_val)}/{format_vital(dbp_val)}, "
                     f"Pain={format_vital(pain_val)}/10")

        case_card = (
            f"<div style='font-size: 14px; line-height: 1.6;'>"
            f"**Case {i+1} (Match: {int(score*100)}%)**<br>"
            f"**Subject:** {subj_id} | **Stay:** {stay_id}<br>"
            f"**Complaint:** {complaint_txt}<br>"
            f"**Vitals:** {vitals_str}<br>"
            f"**Outcome:** {outcome}"
            f"</div>"
            f"<hr style='margin: 10px 0; border: 0; border-top: 1px solid #eee;'>"
        )
        results.append(case_card)
    
    # Return as HTML instead of Markdown for better control
    return "".join(results) if results else "No similar cases found."

def format_risk_badge(text):
    text = str(text).lower()
    if "high" in text:
        return f"<div style='background-color: #ef4444; color: white; padding: 10px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;'>HIGH RISK</div>"
    elif "medium" in text:
        return f"<div style='background-color: #f59e0b; color: white; padding: 10px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;'>MEDIUM RISK</div>"
    else:
        return f"<div style='background-color: #10b981; color: white; padding: 10px; border-radius: 5px; text-align: center; font-size: 20px; font-weight: bold;'>LOW RISK</div>"

def clean_output(text):
    """Removes redundant words like 'Analyze', 'Response:', or 'TESTS:'"""
    text = text.replace("Response:", "").replace("Analyze", "").strip()
    # Remove TESTS: if it appears in reasoning (should only be in tests box)
    text = re.sub(r"TESTS:.*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    # Remove leading bullets or numbers if the model adds them
    text = re.sub(r"^[\-\*\d\.]+\s*", "", text)
    return text

def predict(complaint, temp, hr, rr, o2, sbp, dbp, pain):
    # 1. RAG & Logic
    similar_cases = retrieve_similar_cases(complaint)
    vitals_status = check_vitals_logic(temp, hr, rr, o2, sbp, dbp)
    
    # Generate accurate vital signs context
    vitals_context = validate_vitals_and_add_context(temp, hr, rr, o2, sbp, dbp, pain)
    
    # 2. STRICT PROMPT with vital range context
    prompt = f"""[INST] You are an Emergency Medical Assistant analyzing a patient case.

PATIENT DATA:
Complaint: {complaint}
Vitals: Temperature={temp}°F, Heart Rate={hr}bpm, Respiratory Rate={rr}/min, O2 Saturation={o2}%, Blood Pressure={sbp}/{dbp}mmHg, Pain={pain}/10

ACCURATE VITAL SIGNS ASSESSMENT:
{vitals_context}

CRITICAL ALERTS:
{vitals_status}

STRICT RULES (MUST FOLLOW):
1. NEVER mention age, gender, or demographics (FORBIDDEN: elderly, male, female, he, she, his, her, man, woman, boy, girl, old, young)
2. ONLY use "the patient" or "they/their/them"
3. Start reasoning: "The patient presents with [complaint]."
4. Use the ACCURATE VITAL SIGNS ASSESSMENT above - do NOT reinterpret the values
5. If a vital says "within normal range" - it IS normal, do NOT call it abnormal
6. Do NOT mention: history, past conditions, previous episodes
7. Do NOT list tests in reasoning section
8. Risk probability: HIGH=70-95%, MEDIUM=40-60%, LOW=5-30%

EXAMPLE (CORRECT):
"The patient presents with chest pain. Temperature 98°F is within normal range. Heart rate 110 bpm is above normal range indicating tachycardia..."

FORMAT:
RISK_LEVEL: [High/Medium/Low]
PROBABILITY: [5-95]%
REASONING: [Use the vital signs assessment provided, do not reinterpret]
TESTS: [Test 1, Test 2, Test 3]

[/INST]"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400, 
                temperature=0.1, # Very low temp for strict formatting
                top_p=0.9, 
                do_sample=True,
                repetition_penalty=1.3 
            )
        
        raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        
        # 3. Parsing & Cleaning
        risk_match = re.search(r"RISK_LEVEL:\s*(.*?)(?=\n|PROBABILITY:)", raw_text, re.IGNORECASE)
        risk_val = risk_match.group(1).strip() if risk_match else "Unknown"
        
        prob_match = re.search(r"PROBABILITY:\s*(.*?)(?=\n|REASONING:)", raw_text, re.IGNORECASE)
        probability_raw = prob_match.group(1).strip() if prob_match else "N/A"
        
        # --- FIX: Validate consistency between risk level and probability ---
        # Extract numeric probability
        prob_num_match = re.search(r"(\d+)", probability_raw)
        if prob_num_match:
            prob_value = int(prob_num_match.group(1))
        else:
            prob_value = None
        
        # Fix inconsistencies - ensure risk level matches probability
        # NEVER allow 0% or 100% - range is 5-95%
        if "high" in risk_val.lower():
            # HIGH RISK should have high probability (70-95%)
            if prob_value is None or prob_value < 70:
                probability = "75%"
            elif prob_value > 95:
                probability = "95%"  # Cap at 95%
            else:
                probability = probability_raw
        elif "low" in risk_val.lower():
            # LOW RISK should have low probability (5-30%)
            if prob_value is None or prob_value > 30:
                probability = "25%"
            elif prob_value < 5:
                probability = "5%"  # Floor at 5%
            else:
                probability = probability_raw
        elif "medium" in risk_val.lower():
            # MEDIUM RISK should be 40-60%
            if prob_value is None or prob_value < 40 or prob_value > 60:
                probability = "50%"
            else:
                probability = probability_raw
        else:
            # If risk level is unclear, default to LOW with 25%
            probability = "25%"
        
        risk_html = format_risk_badge(risk_val)

        reason_match = re.search(r"REASONING:\s*(.*?)(?=\nTESTS:|TESTS:|$)", raw_text, re.DOTALL | re.IGNORECASE)
        if reason_match:
            reasoning = clean_output(reason_match.group(1))
            
            # --- POST-PROCESSING: Remove any gender/age terms that slipped through ---
            forbidden_terms = [
                'elderly', 'male', 'female', 'man', 'woman', 'boy', 'girl',
                'he ', 'she ', 'his ', 'her ', 'him ', 'himself', 'herself',
                'Mr.', 'Mrs.', 'Ms.', 'gentleman', 'lady', 'old', 'young',
                'An elderly', 'A male', 'A female', 'The man', 'The woman'
            ]
            
            for term in forbidden_terms:
                # Case-insensitive replacement
                reasoning = re.sub(rf'\b{term}\b', 'the patient', reasoning, flags=re.IGNORECASE)
            
            # Fix doubled "the patient the patient"
            reasoning = re.sub(r'the patient\s+the patient', 'the patient', reasoning, flags=re.IGNORECASE)
            
            # Ensure it starts with "The patient presents"
            if not reasoning.lower().startswith('the patient'):
                reasoning = f"The patient presents with {complaint}. " + reasoning
                
        else:
            reasoning = clean_output(raw_text)
            # Apply same filtering
            for term in ['elderly', 'male', 'female', 'he ', 'she ', 'his ', 'her ']:
                reasoning = re.sub(rf'\b{term}\b', 'the patient', reasoning, flags=re.IGNORECASE)

        tests_match = re.search(r"TESTS:\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)
        if tests_match:
            tests = clean_output(tests_match.group(1))
            # Further cleanup: remove any accidental TESTS: prefix
            tests = tests.replace("TESTS:", "").strip()
            # Remove (Fallback) if present
            tests = tests.replace("(Fallback)", "").strip()
        else:
            # If no tests found, provide defaults WITHOUT showing "Fallback"
            if "chest" in complaint.lower() or "heart" in complaint.lower():
                tests = "ECG, Cardiac Troponin, Chest X-Ray"
            elif "head" in complaint.lower() or "neuro" in complaint.lower():
                tests = "CT Head, Neurological Exam, Blood Glucose"
            else:
                tests = "Complete Blood Count, Basic Metabolic Panel, Urinalysis"
        
        return risk_html, probability, reasoning, tests, similar_cases

    except Exception as e:
        return "Error", "Error", str(e), "Error", "Error"

# ==========================================
# 3. UI LAYOUT
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# GenMedX: AI Healthcare Agent")
    gr.Markdown(f"### Model: {ADAPTER_ID} | RAG: Active")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Patient Data")
            complaint = gr.Textbox(label="Chief Complaint", value="Chest pain radiating to jaw, sweating", lines=3)
            with gr.Row():
                temp = gr.Number(label="Temp (°F)", value=98.6)
                hr = gr.Number(label="HR (bpm)", value=110)
            with gr.Row():
                rr = gr.Number(label="Resp Rate", value=22)
                o2 = gr.Number(label="O2 Sat (%)", value=95)
            with gr.Row():
                sbp = gr.Number(label="Systolic BP", value=145)
                dbp = gr.Number(label="Diastolic BP", value=90)
            pain = gr.Slider(0, 10, value=7, label="Pain Level")
            btn = gr.Button("Analyze Case", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### AI Assessment")
            with gr.Row():
                out_risk = gr.HTML(label="Risk Level")
                out_prob = gr.Textbox(label="Risk Probability", lines=1)
            
            gr.Markdown("#### Clinical Reasoning")
            out_reasoning = gr.Textbox(show_label=False, lines=5)
            
            gr.Markdown("#### Recommended Tests")
            out_tests = gr.Textbox(show_label=False, lines=2)
            
            gr.Markdown("#### Similar Historical Cases (RAG)")
            out_rag = gr.HTML("*(RAG Results will appear here)*")

    btn.click(
        fn=predict,
        inputs=[complaint, temp, hr, rr, o2, sbp, dbp, pain],
        outputs=[out_risk, out_prob, out_reasoning, out_tests, out_rag]
    )

if __name__ == "__main__":
    demo.launch()