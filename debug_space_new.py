from gradio_client import Client
import os

def debug_space():
    client = Client("hamsaram/GenMedX-RAG")
    
    # Test Data
    complaint = "Chest pain radiating to left arm, sweating, nausea. History of hypertension."
    vitals = {
        'temperature': 98.6,
        'heartrate': 110,
        'resprate': 22,
        'o2sat': 96,
        'sbp': 150,
        'dbp': 95,
        'pain': 8
    }
    
    # Construct Prompt (Same as in risk_predictor.py)
    prompt = f"""[INST] You are an expert emergency physician. Analyze this patient case strictly following the rules.

PATIENT DATA:
Complaint: {complaint}
Vitals: Temperature={vitals.get('temperature')}F, Heart Rate={vitals.get('heartrate')}bpm, Respiratory Rate={vitals.get('resprate')}/min, O2 Saturation={vitals.get('o2sat')}%, Blood Pressure={vitals.get('sbp')}/{vitals.get('dbp')}mmHg, Pain={vitals.get('pain')}/10

STRICT RULES:
1. Risk probability: HIGH=70-95%, MEDIUM=40-60%, LOW=5-30%
2. Format: RISK_LEVEL: ..., PROBABILITY: ..., REASONING: ..., TESTS: ...

[/INST]"""

    print("🚀 Sending request to Space...")
    try:
        result = client.predict(
            prompt, # INJECTED PROMPT
            float(vitals.get('temperature')),
            float(vitals.get('heartrate')),
            float(vitals.get('resprate')),
            float(vitals.get('o2sat')),
            float(vitals.get('sbp')),
            float(vitals.get('dbp')),
            float(vitals.get('pain')),
            api_name="/predict"
        )
        
        print("\n✅ Space Response Received!")
        print(f"Type: {type(result)}")
        print(f"Length: {len(result)}")
        
        print("\n--- ITEM 0 (Risk HTML) ---")
        print(result[0])
        
        print("\n--- ITEM 1 (Probability) ---")
        print(result[1])
        
        print("\n--- ITEM 2 (Reasoning) ---")
        print(result[2])
        
        print("\n--- ITEM 3 (Tests) ---")
        print(result[3])
        
        print("\n--- ITEM 4 (RAG) ---")
        print(result[4])
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_space()
