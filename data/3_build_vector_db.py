"""
Step 3: Build vector database for RAG
Generates embeddings and creates Qdrant vector database
"""
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QDRANT_URL, QDRANT_API_KEY, RAG_CONFIG

def load_knowledge_base():
    """Load knowledge base from JSON"""
    print("📂 Loading knowledge base...")
    
    with open("./data/knowledge_base.json") as f:
        knowledge_base = json.load(f)
    
    print(f"✅ Loaded {len(knowledge_base)} Q&A pairs")
    return knowledge_base

def initialize_qdrant():
    """Initialize Qdrant client"""
    print("🔌 Connecting to Qdrant Cloud...")
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌ Error: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        print("   Get your credentials from: https://cloud.qdrant.io")
        return None
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("✅ Connected to Qdrant")
        return client
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return None

def create_collection(client):
    """Create Qdrant collection"""
    collection_name = RAG_CONFIG['collection_name']
    
    print(f"📦 Creating collection '{collection_name}'...")
    
    try:
        # Delete collection if it exists
        try:
            client.delete_collection(collection_name=collection_name)
            print(f"   Deleted existing collection")
        except:
            pass
        
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,  # all-MiniLM-L6-v2 embedding size
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection '{collection_name}' created")
        return True
        
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False

def generate_embeddings_and_upload(client, knowledge_base):
    """Generate embeddings and upload to Qdrant"""
    print("\n🧠 Generating embeddings and uploading...")
    
    # Initialize embedding model
    model_name = RAG_CONFIG['embedding_model']
    print(f"   Loading embedding model: {model_name}")
    encoder = SentenceTransformer(model_name)
    
    collection_name = RAG_CONFIG['collection_name']
    batch_size = 100
    points = []
    
    # Process in batches with progress bar
    for idx, qa in enumerate(tqdm(knowledge_base, desc="Processing")):
        # Create text for embedding
        text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
        
        # Generate embedding
        embedding = encoder.encode(text).tolist()
        
        # Create point
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "question": qa['question'],
                "answer": qa['answer'],
                "source": qa['source'],
                "type": qa.get('type', 'unknown')
            }
        )
        points.append(point)
        
        # Upload batch
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    print(f"\n✅ Uploaded {len(knowledge_base)} documents to vector database")

def test_retrieval(client):
    """Test vector search"""
    print("\n🧪 Testing retrieval...")
    
    # Initialize encoder
    encoder = SentenceTransformer(RAG_CONFIG['embedding_model'])
    
    # Test query
    test_query = "What is PCOS?"
    query_vector = encoder.encode(test_query).tolist()
    
    # Search using the correct method for newer Qdrant versions
    from qdrant_client.models import SearchRequest
    
    results = client.query_points(
        collection_name=RAG_CONFIG['collection_name'],
        query=query_vector,
        limit=3
    ).points
    
    print(f"   Query: '{test_query}'")
    print(f"   Top 3 results:")
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. Score: {result.score:.3f}")
        print(f"      Q: {result.payload['question'][:80]}...")
        print(f"      A: {result.payload['answer'][:80]}...")
        print(f"      Source: {result.payload['source']}")
    
    print("\n✅ Retrieval test successful!")

def main():
    """Build vector database"""
    print("=" * 50)
    print("STEP 3: Building Vector Database")
    print("=" * 50)
    
    # Load knowledge base
    knowledge_base = load_knowledge_base()
    
    # Initialize Qdrant
    client = initialize_qdrant()
    if not client:
        return
    
    # Create collection
    if not create_collection(client):
        return
    
    # Generate embeddings and upload
    generate_embeddings_and_upload(client, knowledge_base)
    
    # Test retrieval
    test_retrieval(client)
    
    print("\n✅ Vector database ready!")
    print(f"   Collection: {RAG_CONFIG['collection_name']}")
    print(f"   Documents: {len(knowledge_base)}")
    print(f"   Embedding model: {RAG_CONFIG['embedding_model']}")

if __name__ == "__main__":
    main()
