import os
from gradio_client import Client
from dotenv import load_dotenv
from config import RAG_CONFIG
from rag.retriever import RAGRetriever

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
            # Don't raise here, allow retry in predict

    def _load_rag(self):
        """Load RAG components"""
        try:
            triage_collection = RAG_CONFIG.get('triage_collection', 'triage_cases')
            self.retriever = RAGRetriever(collection_name=triage_collection)
            print(f"✅ RAG Ready ({triage_collection})")
        except Exception as e:
            print(f"❌ Error loading RAG: {e}")

    def predict(self, complaint, vitals):
        """Generate risk prediction using HF Space"""
        # 1. Get Similar Cases
        similar_cases = self._get_similar_cases(complaint)
        
        # 3. Generate via Space
        try:
            if not self.client:
                self._connect_to_space()
                
            # 2. Construct Prompt (Re-introducing RAG with Chain-of-Thought)
            # We will INJECT this prompt into the 'complaint' field of the Space
            # to force it to follow our logic.
            prompt = self._construct_prompt(complaint, vitals, similar_cases)
            
            print("🧠 Sending request to Space (Prompt Injection)...")
            
            # The Space expects separate arguments.
            # We hijack the 'complaint' argument to send our full prompt.
            # We still send the vitals so the Space doesn't use defaults (just in case).
            
            result = self.client.predict(
                prompt, # INJECTED PROMPT
                float(vitals.get('temperature', 98.6)),
                float(vitals.get('heartrate', 80)),
                float(vitals.get('resprate', 16)),
                float(vitals.get('o2sat', 98)),
                float(vitals.get('sbp', 120)),
                float(vitals.get('dbp', 80)),
                float(vitals.get('pain', 0)),
                api_name="/predict" 
            )
            
            # Result is a tuple from the Gradio Space
            # (Risk HTML, Confidence, Reasoning, Tests, Similar Cases)
            print(f"✅ Space Response: {result}")
            
            # Parse Tuple Output
            risk_html = result[0]  # e.g. <div...>✅ LOW RISK</div> or 🚨 HIGH RISK
            reasoning = result[2]
            tests_str = result[3]
            similar_cases_str = result[4] 
            
            # Extract Risk from HTML
            import re
            risk_match = re.search(r'✅ (.*?)<', risk_html)
            if not risk_match:
                risk_match = re.search(r'⚠️ (.*?)<', risk_html)
            if not risk_match:
                risk_match = re.search(r'🚨 (.*?)<', risk_html) # Handle High Risk icon
            risk = risk_match.group(1) if risk_match else "Unknown"
            
            # Parse Tests
            tests = [t.strip() for t in tests_str.split(',')]
            
            return {
                "risk": risk,
                "reasoning": reasoning,
                "tests": tests,
                "similar_cases": similar_cases # Use our local RAG citations
            }
            
        except Exception as e:
            print(f"❌ Space Inference Error: {e}")
            raise e


        
    def _construct_prompt(self, complaint, vitals, similar_cases):
        """Create the prompt for the model"""
        
        # Format similar cases for context
        context_str = ""
        if similar_cases:
            context_str = "REFERENCE CASES (Historical Data - NOT the current patient):\n"
            for i, case in enumerate(similar_cases[:3]): # Top 3
                # EXTRACT ONLY COMPLAINT AND ACUITY. DO NOT SHOW VITALS OF PAST CASES.
                # case is a dict: {'question': 'Complaint: ...', 'answer': 'Acuity: ... | Vitals: ...'}
                
                # Parse the complaint
                complaint_part = case.get('question', 'N/A')
                
                # Parse the acuity (remove vitals)
                answer_part = case.get('answer', '')
                if '|' in answer_part:
                    acuity_part = answer_part.split('|')[0].strip() # Keep only "Acuity: X"
                else:
                    acuity_part = answer_part
                
                context_str += f"- Case {i+1}: {complaint_part} -> {acuity_part}\n"
        
        return f"""
        [INST] You are an expert emergency physician. Follow the examples below to analyze the patient.
        
        === EXAMPLE 1 (Normal Vitals) ===
        Patient Data:
        - Complaint: Headache
        - Vitals: Temp:98.6F, HR:75, RR:16, BP:120/80
        Reasoning: Patient has HR of 75 (Normal) and BP of 120/80 (Normal). No signs of distress. Headache is likely benign.
        Risk: LOW RISK
        
        === EXAMPLE 2 (Abnormal Vitals) ===
        Patient Data:
        - Complaint: Chest pain
        - Vitals: Temp:98.6F, HR:120, RR:24, BP:85/50
        Reasoning: Patient has HR of 120 (Tachycardia) and BP of 85/50 (Hypotension). This indicates shock. High risk of sepsis or MI.
        Risk: HIGH RISK
        
        ============================================================
        
        === HISTORICAL CONTEXT (For Reference Only) ===
        {context_str}
        ================================================
        
        === TARGET PATIENT (Analyze THIS person) ===
        - Complaint: {complaint}
        - Vitals: 
          * Temp: {vitals.get('temperature')} F
          * HR: {vitals.get('heartrate')} bpm
          * RR: {vitals.get('resprate')} /min
          * O2 Sat: {vitals.get('o2sat')} %
          * BP: {vitals.get('sbp')}/{vitals.get('dbp')} mmHg
          * Pain: {vitals.get('pain')}/10
        ============================================
        
        Task:
        1. Risk Level (High/Low)
        2. Reasoning (3 sentences). START by explicitly stating the patient's vitals and whether they are NORMAL or ABNORMAL.
        3. 3 Diagnostic Tests
        
        Format:
        Risk: [Level]
        Reasoning: [Text]
        Tests: [List]
        [/INST]
        """

    def _parse_response(self, text):
        """Parse the generated text to extract structured data"""
        risk = "Unknown"
        reasoning = "Could not parse reasoning."
        tests = []
        
        # Clean up text (remove prompt)
        if "[/INST]" in text:
            output = text.split("[/INST]")[1].strip()
        else:
            output = text.strip()
            
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("risk:"):
                risk = line.split(":", 1)[1].strip()
            elif line.lower().startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.lower().startswith("tests:"):
                tests_str = line.split(":", 1)[1].strip()
                tests = [t.strip() for t in tests_str.split(',')]
                
        return {
            "risk": risk,
            "reasoning": reasoning,
            "tests": tests,
            "raw_output": output
        }

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
