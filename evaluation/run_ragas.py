"""
RAGAS Evaluation Script
Evaluates all 4 models using RAGAS framework
"""
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,  # Semantic similarity to reference
    context_entity_recall  # Entity-level recall
)
from datasets import Dataset
import json
import pandas as pd
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_manager import ModelManager
from rag.retriever import RAGRetriever
from config import OPENAI_API_KEY

# Set OpenAI API key for RAGAS
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_test_set(limit=50):
    """Load test set"""
    print(f"📂 Loading test set (limit: {limit})...")
    
    with open("./data/test_set.json") as f:
        test_set = json.load(f)
    
    # Limit to specified number
    test_set = test_set[:limit]
    
    print(f"✅ Loaded {len(test_set)} test questions")
    return test_set

def evaluate_model(model_name, test_set, model_manager, rag_retriever):
    """Evaluate a single model using RAGAS"""
    print(f"\n🔬 Evaluating {model_name}...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    errors = []
    
    # Generate answers for all test questions
    for qa in tqdm(test_set, desc=f"Generating {model_name} answers"):
        question = qa['question']
        
        # Retrieve context
        context, results = rag_retriever.retrieve(question)
        
        # Generate answer
        result = model_manager.generate(model_name, question, context, use_rag=True)
        
        if not result.get('error'):
            questions.append(question)
            answers.append(result['answer'])
            contexts.append([context])  # RAGAS expects list of contexts
            ground_truths.append(qa['answer'])
        else:
            # Log error with full details
            error_msg = result.get('answer', 'Unknown error')
            errors.append(f"Q: {question[:50]}... | Error: {error_msg[:100]}")
        
        # Add delay to avoid rate limiting
        import time
        time.sleep(5.0)  # 5s delay between requests to avoid 429 errors
    
    print(f"   Generated {len(answers)} answers")
    
    if errors:
        print(f"   ⚠️  {len(errors)} errors occurred:")
        for i, error in enumerate(errors[:3], 1):  # Show first 3 errors
            print(f"      {i}. {error}")
    
    if len(answers) == 0:
        print(f"   ❌ No valid answers generated. Cannot evaluate.")
        return None
    
    # Create dataset for RAGAS
    eval_dataset = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    })
    
    # Run RAGAS evaluation
    print(f"   Running RAGAS evaluation on {len(answers)} questions...")
    try:
        # Explicitly configure OpenAI LLM for evaluation to fix NaNs
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        
        # Initialize judge LLM
        openai_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
        judge_llm = LangchainLLMWrapper(openai_llm)
        
        result = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,           # No hallucinations
                answer_relevancy,       # Answers the question
                context_precision,      # Retrieved relevant docs
                context_recall,         # Retrieved all needed info
                answer_correctness,     # Overall quality
                answer_similarity,      # Semantic similarity to reference
                context_entity_recall   # Entity-level recall
            ],
            llm=judge_llm  # Pass the explicit judge
        )
        print(f"✅ {model_name} evaluation complete")
        return result
    except Exception as e:
        print(f"❌ RAGAS evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run RAGAS evaluation on all 4 models"""
    print("=" * 50)
    print("RAGAS EVALUATION")
    print("=" * 50)
    
    # Initialize components
    model_manager = ModelManager()
    rag_retriever = RAGRetriever()
    
    # Load test set
    test_set = load_test_set(limit=5)  # Use 5 questions for quick debug
    
    # Evaluate all models
    models = ["gemini", "llama", "biomistral", "meditron"]
    # models = ["biomistral"]  # ONLY run BioMistral as requested to save costs
    results = {}
    
    for model in models:
        try:
            result = evaluate_model(model, test_set, model_manager, rag_retriever)
            results[model] = result
        except Exception as e:
            print(f"❌ Error evaluating {model}: {e}")
            results[model] = None
    
    # Save results
    print("\n💾 Saving results...")
    os.makedirs("./evaluation", exist_ok=True)
    
    # Convert to dict for JSON serialization
    results_dict = {}
    for model, result in results.items():
        if result:
            results_dict[model] = {}
            try:
                # Use to_pandas() which is the most robust way to get scores in RAGAS
                if hasattr(result, 'to_pandas'):
                    df_scores = result.to_pandas()
                    print(f"   📊 Extracted scores for {model} (shape: {df_scores.shape})")
                    
                    for metric_name in ['faithfulness', 'answer_relevancy', 'context_precision', 
                                       'context_recall', 'answer_correctness', 'answer_similarity', 
                                       'context_entity_recall']:
                        if metric_name in df_scores.columns:
                            # Calculate mean and convert to float
                            mean_score = df_scores[metric_name].mean()
                            results_dict[model][metric_name] = float(mean_score)
                        else:
                            print(f"      ⚠️ Metric {metric_name} not found in dataframe")
                else:
                    # Fallback for older versions or dicts
                    print(f"   ⚠️ Result object has no to_pandas(), treating as dict")
                    for metric_name in ['faithfulness', 'answer_relevancy', 'context_precision', 
                                       'context_recall', 'answer_correctness', 'answer_similarity', 
                                       'context_entity_recall']:
                        if metric_name in result:
                            results_dict[model][metric_name] = float(result[metric_name])
            except Exception as e:
                print(f"   ❌ Error parsing results for {model}: {e}")
                import traceback
                traceback.print_exc()
                results_dict[model] = {}
    
    with open("./evaluation/ragas_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print("✅ Results saved to ./evaluation/ragas_results.json")
    
    # Display results
    print("\n" + "=" * 50)
    print("RAGAS RESULTS")
    print("=" * 50)
    
    df = pd.DataFrame(results_dict).T
    print(df.to_string())
    
    # Calculate average scores
    df['average'] = df.mean(axis=1)
    
    # Find winner
    winner = df['average'].idxmax()
    print(f"\n🏆 Winner: {winner}")
    print(f"   Average score: {df.loc[winner, 'average']:.3f}")
    
    # Save comparison table
    df.to_csv("./evaluation/comparison_table.csv")
    print("\n✅ Comparison table saved to ./evaluation/comparison_table.csv")

if __name__ == "__main__":
    main()
