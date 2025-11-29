"""
Step 4: Upload Triage Data to Qdrant
Uploads local MIMIC-IV triage data for "Similar Cases" retrieval
"""
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QDRANT_URL, QDRANT_API_KEY, RAG_CONFIG

def upload_triage_data():
    print("=" * 50)
    print("STEP 4: Uploading Triage Data to Vector DB")
    print("=" * 50)

    # 1. Load Data
    csv_path = "risk_analysis_biomistral/data/triage_raw.csv"
    print(f"📂 Loading data from {csv_path}...")
    
    # Read first 10000 rows to find enough good data
    df = pd.read_csv(csv_path, nrows=10000)
    
    # Filter: Keep only rows with essential vitals (HR, Temp, SBP, O2)
    # This ensures "Similar Cases" are actually useful
    df = df.dropna(subset=['heartrate', 'temperature', 'sbp', 'o2sat'])
    
    # Limit to 5000 high-quality cases
    df = df.head(5000)
    
    print(f"✅ Loaded {len(df)} high-quality cases (filtered out incomplete data)")
    
    # Fill remaining NaNs (like pain or resp rate)
    df = df.fillna("Unknown")

    # 2. Initialize Qdrant
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌ Error: Qdrant credentials missing in .env")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collection_name = RAG_CONFIG.get('triage_collection', 'triage_cases')
    
    # 3. Create Collection
    print(f"📦 Creating collection '{collection_name}'...")
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"✅ Collection created")
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return

    # 4. Generate Embeddings & Upload
    print("\n🧠 Generating embeddings...")
    encoder = SentenceTransformer(RAG_CONFIG['embedding_model'])
    
    points = []
    batch_size = 100
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Create text representation for embedding
        # We want to find similar cases based on Complaint and Vitals
        text = f"Complaint: {row['chiefcomplaint']}. "
        text += f"Vitals: T {row['temperature']}, HR {row['heartrate']}, RR {row['resprate']}, "
        text += f"BP {row['sbp']}/{row['dbp']}, O2 {row['o2sat']}%"
        
        embedding = encoder.encode(text).tolist()
        
        # Create payload (metadata to display)
        payload = {
            "subject_id": str(row['subject_id']),
            "stay_id": str(row['stay_id']),
            "complaint": str(row['chiefcomplaint']),
            "acuity": str(row['acuity']),
            "vitals": {
                "temp": str(row['temperature']),
                "hr": str(row['heartrate']),
                "bp": f"{row['sbp']}/{row['dbp']}",
                "o2": str(row['o2sat'])
            }
        }
        
        points.append(PointStruct(
            id=idx,
            vector=embedding,
            payload=payload
        ))
        
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []
            
    # Upload remaining
    if points:
        client.upsert(collection_name=collection_name, points=points)
        
    print(f"\n✅ Successfully uploaded {len(df)} triage cases to Qdrant!")

if __name__ == "__main__":
    upload_triage_data()
