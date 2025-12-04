from gradio_client import Client

def test_normal_input():
    client = Client("hamsaram/GenMedX-RAG")
    
    complaint = "fever, headache, left leg pain"
    vitals = {
        'temperature': 98.6,
        'heartrate': 80,
        'resprate': 16,
        'o2sat': 98,
        'sbp': 120,
        'dbp': 80,
        'pain': 5
    }
    
    print(f"🚀 Sending NORMAL request (No Injection): '{complaint}'")
    
    try:
        result = client.predict(
            complaint, # NORMAL INPUT
            float(vitals['temperature']),
            float(vitals['heartrate']),
            float(vitals['resprate']),
            float(vitals['o2sat']),
            float(vitals['sbp']),
            float(vitals['dbp']),
            float(vitals['pain']),
            api_name="/predict"
        )
        
        print("\n✅ Response Received!")
        print("--- RAG HTML (Item 4) ---")
        print(result[4][:500]) # Print first 500 chars to check match %
        
        print("\n--- Reasoning (Item 2) ---")
        print(result[2])
        
        print("\n--- Tests (Item 3) ---")
        print(result[3])
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_normal_input()
