from gradio_client import Client
import re
import time
from backend.metrics import metrics_tracker

class SymptomPredictor:
    def __init__(self):
        # Note: This Space must be RUNNING for this to work.
        # If it is paused, the initialization or prediction will fail.
        self.space_id = "Sugandha-Chauhan/biomistral-symptom-diagnosis-v2"
        self.client = None

    def _get_client(self):
        if self.client is None:
            try:
                self.client = Client(self.space_id)
            except Exception as e:
                print(f"Error connecting to Space {self.space_id}: {e}")
                raise e
        return self.client

    def predict(self, symptoms: str):
        """
        Predicts diagnosis and generates explanation from symptoms.
        Returns a dictionary with structured data.
        """
        start_time = time.time()
        success = False
        error_msg = None
        
        client = self._get_client()
        
        try:
            # The app.py shows 1 input (symptoms) and 4 outputs
            # usually mapped to the default api_name='/predict'
            result = client.predict(
                symptoms,
                api_name="/diagnose_and_explain_interface"
            )
            
            # Result tuple: (diagnosis_md, confidence_md, symptoms_md, explanation_md)
            # Example:
            # 0: "**Diagnosis:** Acute Bronchitis"
            # 1: "**Confidence:** 98.5%"
            # 2: "**Symptoms:** cough, fever"
            # 3: "Acute bronchitis is..."
            
            diagnosis_raw = result[0]
            confidence_raw = result[1]
            explanation = result[3]
            
            # Clean up Markdown to get raw values
            diagnosis = diagnosis_raw.replace("**Diagnosis:**", "").strip()
            confidence = confidence_raw.replace("**Confidence:**", "").strip()
            
            success = True
            return {
                "diagnosis": diagnosis,
                "confidence": confidence,
                "explanation": explanation,
                "raw_symptoms": symptoms
            }
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            success = False
            error_msg = str(e)
            return {
                "error": str(e)
            }
        finally:
            duration = time.time() - start_time
            metrics_tracker.log_inference("Symptom Checker", duration, success, error_msg)

# Singleton instance
_predictor = None

def get_symptom_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SymptomPredictor()
    return _predictor
