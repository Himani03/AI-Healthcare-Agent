from cog import BasePredictor, Input, Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("🔄 Loading BioMistral + Adapters...")
        
        # Paths
        self.base_model_id = "BioMistral/BioMistral-7B"
        self.adapter_path = "risk_analysis_biomistral/model/biomistral_finetuned_3epochs"
        
        # Load Base Model
        # device_map="auto" can fail in some containers, explicit cuda is safer for single-GPU
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map="cuda", 
            trust_remote_code=True
        )
        
        # Load Adapters
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        print("✅ Model loaded successfully!")

    def predict(
        self,
        complaint: str = Input(description="Patient Chief Complaint"),
        vitals: str = Input(description="Patient Vitals (formatted string)", default="Unknown")
    ) -> str:
        """Run a single prediction on the model"""
        
        # Construct Prompt
        prompt = f"""
        [INST] You are an expert medical AI. Analyze this patient strictly based on the provided data.
        
        Complaint: {complaint}
        Vitals: {vitals}
        
        Task:
        1. Risk Level (High/Low)
        2. Reasoning (1 sentence). ONLY mention abnormalities present in the vitals above.
        3. 3 Diagnostic Tests
        
        Format:
        Risk: [Level]
        Reasoning: [Text]
        Tests: [List]
        [/INST]
        """
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
