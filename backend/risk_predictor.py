import os
import re
from gradio_client import Client
from dotenv import load_dotenv
from config import RAG_CONFIG
from rag.retriever import RAGRetriever
from backend.metrics import metrics_tracker
import time

load_dotenv()

class RiskPredictor:
    def __init__(self):
        print("🚀 Initializing Risk Predictor (Hugging Face Space)...")
        
        # Configuration
        self.space_id = "hamsaram/GenMedX-RAG"
        self.client = None
        self.retriever = None
        
        # Load Resources
        self._connect_to_space()
        self._load_rag()

    def _connect_to_space(self):
        """Connect to Hugging Face Space"""
        try:
            print(f"⏳ Connecting to Space: {self.space_id}...")
            self.client = Client(self.space_id)
            print("✅ Connected to Space!")
        except Exception as e:
            print(f"❌ Error connecting to Space: {e}")

    def _load_rag(self):
        """Load RAG components"""
        try:
            triage_collection = RAG_CONFIG.get('triage_collection', 'triage_cases')
            self.retriever = RAGRetriever(collection_name=triage_collection)
            print(f"✅ RAG Ready ({triage_collection})")
        except Exception as e:
            print(f"❌ Error loading RAG: {e}")

    def _check_vitals(self, vitals):
        """Generate accurate vital signs assessment (Matching hamsa new logic)"""
        checks = []
        status = []
        
        # HR
        hr = float(vitals.get('heartrate', 80))
        if hr < 60:
            checks.append(f"Heart Rate {hr} bpm is BELOW normal range (bradycardia)")
            status.append(f"ABNORMAL VITALS: Bradycardia (HR {hr})")
        elif hr > 100:
            checks.append(f"Heart Rate {hr} bpm is ABOVE normal range (tachycardia)")
            status.append(f"ABNORMAL VITALS: Tachycardia (HR {hr})")
        else:
            checks.append(f"Heart Rate {hr} bpm is within normal range (60-100)")

        # BP
        sbp = float(vitals.get('sbp', 120))
        dbp = float(vitals.get('dbp', 80))
        if sbp < 90 or dbp < 60:
            checks.append(f"BP {sbp}/{dbp} mmHg is BELOW normal range (hypotension)")
            status.append(f"CRITICAL: Hypotension ({sbp}/{dbp})")
        elif sbp > 140 or dbp > 90:
            checks.append(f"BP {sbp}/{dbp} mmHg is ABOVE normal range (hypertension)")
            status.append(f"ABNORMAL VITALS: Hypertension ({sbp}/{dbp})")
        else:
            checks.append(f"BP {sbp}/{dbp} mmHg is within normal range")

        # O2
        o2 = float(vitals.get('o2sat', 98))
        if o2 < 95:
            checks.append(f"O2 Saturation {o2}% is BELOW normal range (hypoxia)")
            status.append(f"CRITICAL: Hypoxia (O2 {o2}%)")
        else:
            checks.append(f"O2 Saturation {o2}% is within normal range (>95%)")

        # Temp
        temp = float(vitals.get('temperature', 98.6))
        if temp > 100.4:
            checks.append(f"Temperature {temp} F is ABOVE normal range (fever)")
            status.append(f"ABNORMAL VITALS: Fever ({temp} F)")
        elif temp < 95:
            checks.append(f"Temperature {temp} F is BELOW normal range (hypothermia)")
            status.append(f"CRITICAL: Hypothermia ({temp} F)")
        else:
            checks.append(f"Temperature {temp} F is within normal range")

        # RR
        rr = float(vitals.get('resprate', 16))
        if rr > 20:
            checks.append(f"Respiratory Rate {rr}/min is ABOVE normal range (tachypnea)")
            status.append(f"ABNORMAL VITALS: Tachypnea (RR {rr})")
        elif rr < 12:
            checks.append(f"Respiratory Rate {rr}/min is BELOW normal range (bradypnea)")
            status.append(f"ABNORMAL VITALS: Bradypnea (RR {rr})")
        else:
            checks.append(f"Respiratory Rate {rr}/min is within normal range")

        return "\n".join(checks), "\n".join(status)

    def _construct_prompt(self, complaint, vitals):
        """Construct strict prompt with vital checks"""
        vitals_context, vitals_status = self._check_vitals(vitals)
        
        return f"""[INST] You are an expert emergency physician. Analyze this patient case strictly following the rules.

PATIENT DATA:
Complaint: {complaint}
Vitals: Temperature={vitals.get('temperature')}F, Heart Rate={vitals.get('heartrate')}bpm, Respiratory Rate={vitals.get('resprate')}/min, O2 Saturation={vitals.get('o2sat')}%, Blood Pressure={vitals.get('sbp')}/{vitals.get('dbp')}mmHg, Pain={vitals.get('pain')}/10

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
"The patient presents with chest pain. Temperature 98F is within normal range. Heart rate 110 bpm is above normal range indicating tachycardia..."

FORMAT:
RISK_LEVEL: [High/Medium/Low]
PROBABILITY: [5-95]%
REASONING: [Use the vital signs assessment provided, do not reinterpret]
TESTS: [Test 1, Test 2, Test 3]

[/INST]"""

    def _format_rag_context(self, similar_cases):
        """Format RAG results as HTML for frontend"""
        if not similar_cases:
            return "<em>No similar cases found.</em>"
            
        html = "<div style='font-size: 0.9rem;'>"
        for i, case in enumerate(similar_cases[:3]):
            # Parse citation
            complaint = case.get('question', 'N/A').replace("Complaint:", "").strip()
            details = case.get('answer', 'N/A')
            
            # Clean HTML construction (single line to avoid indentation issues)
            html += f"<div style='margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-left: 3px solid #1f77b4; border-radius: 4px;'><strong>Case {i+1}:</strong> {complaint}<br><span style='color: #666; font-size: 0.8rem;'>{details[:100]}...</span></div>"
            
        html += "</div>"
        return html

    def predict(self, complaint, vitals):
        """Generate risk prediction using HF Space"""
        start_time = time.time()
        success = False
        error_msg = None
        
        # 1. Get Similar Cases (RAG)
        similar_cases = self._get_similar_cases(complaint)
        rag_html = self._format_rag_context(similar_cases)
        
        try:
            if not self.client:
                self._connect_to_space()
            
            # Check if connection was successful
            if not self.client:
                raise ConnectionError("Could not connect to Hugging Face Space. Please try again later.")

            # 2. Construct Prompt (Legacy - Not used for Space inference to preserve RAG quality)
            # prompt = self._construct_prompt(complaint, vitals)
            print("🧠 Sending request to Space (Normal Input)...")
            
            # 3. Call Space
            result = self.client.predict(
                complaint, # NORMAL INPUT (Fixes RAG)
                float(vitals.get('temperature', 98.6)),
                float(vitals.get('heartrate', 80)),
                float(vitals.get('resprate', 16)),
                float(vitals.get('o2sat', 98)),
                float(vitals.get('sbp', 120)),
                float(vitals.get('dbp', 80)),
                float(vitals.get('pain', 0)),
                api_name="/predict" 
            )
            
            # 4. Parse Output
            # Space returns: (Risk HTML, Probability, Reasoning, Tests, RAG HTML)
            risk_html = result[0]
            probability = result[1]
            reasoning = result[2]
            tests_str = result[3]
            rag_html_space = result[4]
            
            print(f"🔍 RAW MODEL OUTPUT:\nRisk: {risk_html}\nProb: {probability}\nReasoning: {reasoning}\nTests: {tests_str}\n{'='*50}") 
            
            # Extract Risk Level from HTML
            risk_val = "Unknown"
            clean_html = re.sub(r'<[^>]+>', '', risk_html).strip()
            if "HIGH" in clean_html.upper():
                risk_val = "High Risk"
            elif "MEDIUM" in clean_html.upper():
                risk_val = "Medium Risk"
            elif "LOW" in clean_html.upper():
                risk_val = "Low Risk"
            else:
                risk_val = clean_html
            
            # Parse Tests
            if isinstance(tests_str, str):
                tests = [t.strip() for t in tests_str.split(',')]
            else:
                tests = []

            # Format Space's RAG HTML (Convert Markdown bold to HTML bold)
            if rag_html_space:
                rag_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', rag_html_space)
            else:
                rag_html = "<em>No similar cases found.</em>"
            
            success = True
            return {
                "risk": risk_val,
                "probability": probability,
                "reasoning": reasoning,
                "tests": tests,
                "rag_html": rag_html 
            }
            
        except Exception as e:
            print(f"❌ Space Inference Error: {e}")
            success = False
            error_msg = str(e)
            return {
                "risk": "Error",
                "probability": "N/A",
                "reasoning": f"Error during inference: {str(e)}",
                "tests": [],
                "rag_html": rag_html
            }
        finally:
            duration = time.time() - start_time
            metrics_tracker.log_inference("Risk Analysis", duration, success, error_msg)

    def _get_similar_cases(self, complaint):
        """Retrieve similar cases using RAG"""
        if not self.retriever:
            return []
        try:
            _, results = self.retriever.retrieve(complaint, top_k=3)
            return self.retriever.get_citations(results)
        except Exception as e:
            print(f"RAG Error: {e}")
            return []

# Singleton instance
risk_predictor = None

def get_risk_predictor():
    global risk_predictor
    if risk_predictor is None:
        risk_predictor = RiskPredictor()
    return risk_predictor
