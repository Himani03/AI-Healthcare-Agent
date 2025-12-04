"""
RAG Retriever: Handles vector search and context retrieval
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QDRANT_URL, QDRANT_API_KEY, RAG_CONFIG

class RAGRetriever:
    """Retrieves relevant context from vector database"""
    
    def __init__(self, collection_name=None):
        """Initialize retriever"""
        print("🔍 Initializing RAG Retriever...")
        
        # Connect to Qdrant
        # Runtime fallback for Streamlit Cloud secrets
        url = QDRANT_URL
        key = QDRANT_API_KEY
        
        if hasattr(st, "secrets"):
            if not url and "QDRANT_URL" in st.secrets:
                url = st.secrets["QDRANT_URL"]
                print("   ✅ Found Qdrant URL in st.secrets")
            if not key and "QDRANT_API_KEY" in st.secrets:
                key = st.secrets["QDRANT_API_KEY"]
                print("   ✅ Found Qdrant Key in st.secrets")

        if url and key:
            try:
                self.client = QdrantClient(url=url, api_key=key)
                print(f"   ✅ Connected to Qdrant")
            except Exception as e:
                self.client = None
                print(f"   ⚠️  Qdrant connection failed: {e}")
        else:
            self.client = None
            print("   ⚠️  Qdrant not configured")
        
        # Load embedding model
        # Force CPU to avoid "meta tensor" errors on Streamlit Cloud with accelerate installed
        self.encoder = SentenceTransformer(RAG_CONFIG['embedding_model'], device="cpu")
        print(f"   ✅ Loaded embedding model: {RAG_CONFIG['embedding_model']}")
        
        self.collection_name = collection_name or RAG_CONFIG['collection_name']
        self.top_k = RAG_CONFIG['top_k']
        self.top_k = RAG_CONFIG['top_k']
    
    def retrieve(self, query, top_k=None):
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve (default from config)
        
        Returns:
            tuple: (formatted_context, raw_results)
        """
        if not self.client:
            return "RAG not configured", []
        
        if top_k is None:
            top_k = self.top_k
        
        try:
            # Generate query embedding
            query_vector = self.encoder.encode(query).tolist()
            
            # Search vector database using newer API
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            ).points
            
            # Format context
            context = self._format_context(results)
            
            return context, results
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return f"Error: {str(e)}", []
    
    def _format_context(self, results):
        """Format retrieved documents into context string"""
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            payload = result.payload
            
            # Handle Triage Data Schema
            if 'complaint' in payload:
                vitals = payload.get('vitals', {})
                # Filter out "Unknown" or empty vitals for cleaner display
                vitals_str = ", ".join([f"{k}: {v}" for k, v in vitals.items() if v and str(v).lower() != 'unknown'])
                
                context_parts.append(
                    f"[Case {i}] (Acuity: {payload.get('acuity', 'Unknown')})\n"
                    f"Complaint: {payload.get('complaint')}\n"
                    f"Vitals: {vitals_str}\n"
                )
            # Handle Medical QA Schema
            else:
                context_parts.append(
                    f"[Source {i}] ({payload.get('source', 'Unknown')})\n"
                    f"Q: {payload.get('question', '')}\n"
                    f"A: {payload.get('answer', '')}\n"
                )
        
        return "\n".join(context_parts)
    
    def get_citations(self, results):
        """Extract citations from results"""
        citations = []
        for result in results:
            payload = result.payload
            
            # Handle Triage Data Schema
            if 'complaint' in payload:
                citations.append({
                    "question": f"Complaint: {payload.get('complaint')}",
                    "answer": f"Acuity: {payload.get('acuity')} | Vitals: {payload.get('vitals')}",
                    "source": f"MIMIC-IV Triage (Subject {payload.get('subject_id')})",
                    "score": result.score
                })
            # Handle Medical QA Schema
            else:
                citations.append({
                    "question": payload.get('question', ''),
                    "answer": payload.get('answer', ''),
                    "source": payload.get('source', 'Unknown'),
                    "score": result.score
                })
        return citations

# Test the retriever
if __name__ == "__main__":
    retriever = RAGRetriever()
    
    # Test queries
    test_queries = [
        "What is PCOS?",
        "What can I take for PCOS?",
        "How does metformin work?"
    ]
    
    print("\n🧪 Testing retrieval...")
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        context, results = retriever.retrieve(query, top_k=3)
        
        if results:
            print(f"Retrieved {len(results)} documents:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result.score:.3f}")
                print(f"   Q: {result.payload['question'][:60]}...")
                print(f"   Source: {result.payload['source']}")
        else:
            print("No results found")
