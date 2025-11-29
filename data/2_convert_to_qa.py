"""
Step 2: Convert datasets to conversational Q&A format
Converts MCQ format to simple question-answer pairs for RAG
"""
from datasets import load_from_disk
import json
import os

def convert_medqa_to_qa(dataset):
    """Convert MedQA MCQ format to Q&A pairs"""
    print("🔄 Converting MedQA to Q&A format...")
    
    qa_pairs = []
    
    for item in dataset['train']:
        # Extract question
        question = item['question'].strip()
        
        # Get options and correct answer
        options = item['options']
        answer_idx = item['answer_idx']
        
        # Handle different formats
        if isinstance(options, dict):
            # Options are in dict format (e.g., {'A': 'text', 'B': 'text'})
            # answer_idx is the key (e.g., 'A', 'B', 'C', 'D')
            answer = options.get(answer_idx, '')
        elif isinstance(options, list):
            # Options are in list format
            # Convert answer_idx to integer if it's a string
            if isinstance(answer_idx, str):
                # Map 'A' -> 0, 'B' -> 1, etc.
                idx = ord(answer_idx.upper()) - ord('A')
            else:
                idx = answer_idx
            answer = options[idx] if idx < len(options) else ''
        else:
            answer = str(options)
        
        # Create Q&A pair
        qa_pairs.append({
            "question": question,
            "answer": answer.strip(),
            "source": "MedQA-USMLE",
            "type": "multiple_choice"
        })
    
    print(f"✅ Converted {len(qa_pairs)} MedQA questions")
    return qa_pairs

def convert_pubmedqa_to_qa(dataset):
    """Convert PubMedQA to Q&A pairs"""
    print("🔄 Converting PubMedQA to Q&A format...")
    
    qa_pairs = []
    
    for item in dataset['train']:
        # Extract question
        question = item['question'].strip()
        
        # Get answer (yes/no/maybe) and context
        final_decision = item.get('final_decision', 'unknown')
        
        # Get long answer if available
        long_answer = item.get('long_answer', '')
        
        # Create comprehensive answer
        if long_answer:
            answer = f"{final_decision.capitalize()}. {long_answer}"
        else:
            answer = final_decision.capitalize()
        
        # Create Q&A pair
        qa_pairs.append({
            "question": question,
            "answer": answer.strip(),
            "source": "PubMedQA",
            "type": "yes_no_maybe"
        })
    
    print(f"✅ Converted {len(qa_pairs)} PubMedQA questions")
    return qa_pairs

def split_data(qa_pairs, train_ratio=0.9):
    """Split data into knowledge base and test set"""
    print(f"📊 Splitting data ({train_ratio*100}% knowledge base, {(1-train_ratio)*100}% test)...")
    
    split_idx = int(len(qa_pairs) * train_ratio)
    
    knowledge_base = qa_pairs[:split_idx]
    test_set = qa_pairs[split_idx:]
    
    print(f"   - Knowledge base: {len(knowledge_base)} Q&A pairs")
    print(f"   - Test set: {len(test_set)} questions")
    
    return knowledge_base, test_set

def main():
    """Convert datasets and split into knowledge base and test set"""
    print("=" * 50)
    print("STEP 2: Converting to Q&A Format")
    print("=" * 50)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    medqa = load_from_disk("./data/medqa")
    pubmedqa = load_from_disk("./data/pubmedqa")
    
    # Convert to Q&A format
    medqa_qa = convert_medqa_to_qa(medqa)
    pubmedqa_qa = convert_pubmedqa_to_qa(pubmedqa)
    
    # Combine all Q&A pairs
    all_qa_pairs = medqa_qa + pubmedqa_qa
    print(f"\n📚 Total Q&A pairs: {len(all_qa_pairs)}")
    
    # Split into knowledge base and test set
    knowledge_base, test_set = split_data(all_qa_pairs, train_ratio=0.9)
    
    # Save to JSON files
    print("\n💾 Saving to files...")
    os.makedirs("./data", exist_ok=True)
    
    with open("./data/knowledge_base.json", "w") as f:
        json.dump(knowledge_base, f, indent=2)
    print(f"   ✅ Saved knowledge_base.json ({len(knowledge_base)} pairs)")
    
    with open("./data/test_set.json", "w") as f:
        json.dump(test_set, f, indent=2)
    print(f"   ✅ Saved test_set.json ({len(test_set)} questions)")
    
    # Print sample
    print("\n📋 Sample Q&A pair:")
    sample = knowledge_base[0]
    print(f"   Question: {sample['question'][:100]}...")
    print(f"   Answer: {sample['answer'][:100]}...")
    print(f"   Source: {sample['source']}")
    
    print("\n✅ Conversion complete!")

if __name__ == "__main__":
    main()
