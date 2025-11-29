"""
Step 1: Download datasets from HuggingFace
Downloads MedQA-USMLE and PubMedQA datasets
"""
from datasets import load_dataset
import os

def download_medqa():
    """Download MedQA-USMLE-4-options dataset"""
    print("📥 Downloading MedQA-USMLE-4-options...")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
        
        # Create data directory
        os.makedirs("./data/medqa", exist_ok=True)
        
        # Save to disk
        dataset.save_to_disk("./data/medqa")
        
        print(f"✅ MedQA downloaded: {len(dataset['train'])} questions")
        return dataset
        
    except Exception as e:
        print(f"❌ Error downloading MedQA: {e}")
        return None

def download_pubmedqa():
    """Download PubMedQA dataset"""
    print("📥 Downloading PubMedQA...")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        
        # Create data directory
        os.makedirs("./data/pubmedqa", exist_ok=True)
        
        # Save to disk
        dataset.save_to_disk("./data/pubmedqa")
        
        print(f"✅ PubMedQA downloaded: {len(dataset['train'])} questions")
        return dataset
        
    except Exception as e:
        print(f"❌ Error downloading PubMedQA: {e}")
        return None

def main():
    """Download both datasets"""
    print("=" * 50)
    print("STEP 1: Downloading Medical Datasets")
    print("=" * 50)
    
    # Download MedQA
    medqa = download_medqa()
    
    # Download PubMedQA
    pubmedqa = download_pubmedqa()
    
    if medqa and pubmedqa:
        print("\n✅ All datasets downloaded successfully!")
        print(f"   - MedQA: {len(medqa['train'])} questions")
        print(f"   - PubMedQA: {len(pubmedqa['train'])} questions")
        print(f"   - Total: {len(medqa['train']) + len(pubmedqa['train'])} questions")
    else:
        print("\n❌ Some datasets failed to download")

if __name__ == "__main__":
    main()
