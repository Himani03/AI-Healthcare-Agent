from backend.risk_predictor import RiskPredictor

def test_prediction():
    print("🚀 Testing Risk Predictor...")
    predictor = RiskPredictor()
    
    # User's exact inputs
    complaint = "headache, chest pain, fever"
    vitals = {
        "temperature": 100.0,
        "heartrate": 80,
        "resprate": 16,
        "o2sat": 98,
        "sbp": 120,
        "dbp": 80,
        "pain": 0
    }
    
    print(f"📥 Input: {complaint}")
    print(f"📊 Vitals: {vitals}")
    
    try:
        result = predictor.predict(complaint, vitals)
        print("\n✅ Result:")
        print(f"Risk: {result['risk']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Tests: {result['tests']}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_prediction()
