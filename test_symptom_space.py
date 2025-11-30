from gradio_client import Client

def test_space(space_id):
    print(f"🚀 Testing connection to: {space_id}...")
    try:
        client = Client(space_id)
        print(f"✅ Success! Connected to {space_id}")
        print("--- API Information ---")
        client.view_api()
        print("-----------------------")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to {space_id}: {e}")
        return False

if __name__ == "__main__":
    # Potential Space IDs based on the user's folder/model name
    space_ids = [
        "Sugandha-Chauhan/biomistral-symptom-diagnosis-v2"
    ]
    
    for space_id in space_ids:
        if test_space(space_id):
            break
