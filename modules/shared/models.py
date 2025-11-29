"""
Model Manager: Unified interface for all 4 LLM models
Handles Gemini, Llama, BioMistral, and Meditron
"""
import google.generativeai as genai
import replicate
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GOOGLE_API_KEY, 
    REPLICATE_API_TOKEN, 
    MODELS, 
    GENERATION_PARAMS,
    RAG_PROMPT_TEMPLATE
)

class ModelManager:
    """Manages all 4 LLM models with unified interface"""
    
    def __init__(self):
        """Initialize all models"""
        print("🤖 Initializing Model Manager...")
        
        # Configure Gemini
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.gemini = genai.GenerativeModel(MODELS['gemini']['model_id'])
            print("   ✅ Gemini 1.5 Flash ready")
        else:
            self.gemini = None
            print("   ⚠️  Gemini not configured (missing API key)")
        
        # Configure Replicate
        if REPLICATE_API_TOKEN:
            os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
            print("   ✅ Replicate configured")
            print("   ✅ Llama 3.1 8B ready")
            print("   ✅ BioMistral 7B ready")
            print("   ✅ Meditron 7B ready")
        else:
            print("   ⚠️  Replicate not configured (missing API token)")
    
    def generate(self, model_name, question, context="", use_rag=True):
        """
        Generate answer from specified model
        
        Args:
            model_name: One of ['gemini', 'llama', 'biomistral', 'meditron']
            question: User's question
            context: Retrieved context from RAG (optional)
            use_rag: Whether to use RAG context
        
        Returns:
            dict with answer, model info, and metadata
        """
        # Build prompt
        if use_rag and context:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        # Route to appropriate model
        if model_name == "gemini":
            return self._generate_gemini(prompt, question)
        elif model_name == "llama":
            return self._generate_llama(prompt, question)
        elif model_name == "biomistral":
            return self._generate_biomistral(prompt, question)
        elif model_name == "meditron":
            return self._generate_meditron(prompt, question)
        else:
            return {
                "answer": f"Unknown model: {model_name}",
                "model": model_name,
                "error": True
            }
    
    def _generate_gemini(self, prompt, question):
        """Generate from Gemini"""
        if not self.gemini:
            return {"answer": "Gemini not configured", "model": "gemini", "error": True}
        
        try:
            response = self.gemini.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=GENERATION_PARAMS['temperature'],
                    max_output_tokens=GENERATION_PARAMS['max_tokens'],
                    top_p=GENERATION_PARAMS['top_p']
                )
            )
            
            # Safely get text from response
            if response.parts:
                answer = response.text
            elif response.candidates and response.candidates[0].content.parts:
                answer = response.candidates[0].content.parts[0].text
            else:
                # Handle safety blocks or empty responses
                if response.prompt_feedback:
                    answer = f"Blocked: {response.prompt_feedback}"
                else:
                    answer = "Error: Empty response from Gemini"
            
            return {
                "answer": answer,
                "model": "gemini",
                "model_name": MODELS['gemini']['name'],
                "error": False
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "model": "gemini",
                "error": True
            }
    
    def _generate_llama(self, prompt, question):
        """Generate from Llama 3 8B"""
        try:
            # Use specific version if available
            model_id = f"{MODELS['llama']['model_id']}:{MODELS['llama']['version']}" if 'version' in MODELS['llama'] else MODELS['llama']['model_id']
            
            output = replicate.run(
                model_id,
                input={
                    "prompt": prompt,
                    "max_tokens": GENERATION_PARAMS['max_tokens'],
                    "temperature": GENERATION_PARAMS['temperature'],
                    "top_p": GENERATION_PARAMS['top_p']
                }
            )
            
            # Replicate returns an iterator
            answer = "".join(output)
            
            return {
                "answer": answer,
                "model": "llama",
                "model_name": MODELS['llama']['name'],
                "error": False
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "model": "llama",
                "error": True
            }
    
    def _generate_biomistral(self, prompt, question):
        """Generate from Mistral 7B"""
        try:
            model_id = f"{MODELS['biomistral']['model_id']}:{MODELS['biomistral']['version']}" if 'version' in MODELS['biomistral'] else MODELS['biomistral']['model_id']
            
            output = replicate.run(
                model_id,
                input={
                    "prompt": prompt,
                    "max_tokens": GENERATION_PARAMS['max_tokens'],
                    "temperature": GENERATION_PARAMS['temperature'],
                    "top_p": GENERATION_PARAMS['top_p']
                }
            )
            
            answer = "".join(output)
            
            return {
                "answer": answer,
                "model": "biomistral",
                "model_name": MODELS['biomistral']['name'],
                "error": False
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "model": "biomistral",
                "error": True
            }
    
    def _generate_meditron(self, prompt, question):
        """Generate from Llama 3 70B"""
        try:
            model_id = f"{MODELS['meditron']['model_id']}:{MODELS['meditron']['version']}" if 'version' in MODELS['meditron'] else MODELS['meditron']['model_id']
            
            output = replicate.run(
                model_id,
                input={
                    "prompt": prompt,
                    "max_tokens": GENERATION_PARAMS['max_tokens'],
                    "temperature": GENERATION_PARAMS['temperature'],
                    "top_p": GENERATION_PARAMS['top_p']
                }
            )
            
            answer = "".join(output)
            
            return {
                "answer": answer,
                "model": "meditron",
                "model_name": MODELS['meditron']['name'],
                "error": False
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "model": "meditron",
                "error": True
            }
    
    def get_available_models(self):
        """Get list of available models"""
        return list(MODELS.keys())
    
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        return MODELS.get(model_name, {})

# Test the model manager
if __name__ == "__main__":
    manager = ModelManager()
    
    # Test question
    test_question = "What is PCOS?"
    test_context = "PCOS (Polycystic Ovary Syndrome) is a hormonal disorder common among women of reproductive age."
    
    print("\n🧪 Testing models...")
    for model in manager.get_available_models():
        print(f"\n--- Testing {model} ---")
        result = manager.generate(model, test_question, test_context, use_rag=True)
        if not result.get('error'):
            print(f"Answer: {result['answer'][:100]}...")
        else:
            print(f"Error: {result['answer']}")
