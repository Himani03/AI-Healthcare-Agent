import json
import os
from datasets import Dataset

def format_mistral_instruction(question, answer):
    """Format Q&A pair into Mistral instruction format"""
    # BioMistral uses the standard Mistral chat template
    # <s>[INST] Instruction [/INST] Model answer</s>
    return f"<s>[INST] {question} [/INST] {answer}</s>"

def main():
    print("🔄 Preparing dataset for BioMistral fine-tuning...")
    
    # Load knowledge base
    input_path = "../data/knowledge_base.json"
    output_path = "medical_chat_dataset.jsonl"
    
    if not os.path.exists(input_path):
        print(f"❌ Error: Could not find {input_path}")
        print("   Please run 'python data/2_convert_to_qa.py' first.")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"   Loaded {len(data)} Q&A pairs")
    
    # Format data
    formatted_data = []
    for item in data:
        text = format_mistral_instruction(item['question'], item['answer'])
        formatted_data.append({"text": text})
    
    # Save as JSONL
    with open(output_path, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"✅ Saved formatted dataset to {output_path}")
    print(f"   Sample: {formatted_data[0]['text'][:100]}...")

if __name__ == "__main__":
    main()
