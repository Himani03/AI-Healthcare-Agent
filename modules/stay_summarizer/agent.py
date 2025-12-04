# agent.py
# ML-powered summarizer agent

import os
import pandas as pd
import time
import sys
from typing import List, Tuple, Any
from .ml_model import SummarizerML

# Add root directory to path to import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from backend.metrics import metrics_tracker

# Load dataset
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, "data", "summarizer_data_50_patients.csv")

try:
    patient_data_df = pd.read_csv(CSV_PATH)
    # Normalize columns
    if "subject_id" in patient_data_df.columns:
        patient_data_df["subject_id"] = pd.to_numeric(patient_data_df["subject_id"], errors="coerce")
    print(f"✅ Loaded CSV '{CSV_PATH}' with {len(patient_data_df):,} rows.")
except Exception as e:
    print(f"❌ Error loading CSV '{CSV_PATH}': {e}")
    patient_data_df = pd.DataFrame()

def get_available_patient_ids() -> List[int]:
    """Return a list of unique patient IDs available in the dataset."""
    if patient_data_df.empty:
        return []
    return sorted(patient_data_df["subject_id"].dropna().unique().astype(int).tolist())

def run_agent(patient_id: int) -> str:
    """
    Summarize all stays for a patient.
    """
    if patient_data_df.empty:
        return "Error: Patient data not loaded."

    # Filter by patient ID
    rows = patient_data_df[patient_data_df["subject_id"] == patient_id]
    if rows.empty:
        return f"No records found for Patient ID {patient_id}."

    # Group by stay_id
    grouped_stays = list(rows.groupby("stay_id", dropna=True))
    
    summarizer = SummarizerML()
    outputs = []

    print(f"--- AGENT: Summarizing {len(grouped_stays)} stays for Patient {patient_id} ---")
    
    start_time = time.time()
    success = True
    error_msg = None

    try:
        for _stay_id, stay_df in grouped_stays:
            # Extract date for header
            intime = ""
            try:
                if "intime" in stay_df.columns:
                    intime_raw = str(stay_df["intime"].iloc[0])
                    intime = intime_raw.split(" ")[0]
            except Exception:
                pass

            # Generate summary
            summary_text = summarizer.summarize_stay(stay_df)
            
            header = f"**Stay on {intime}**" if intime else "**Hospital Stay**"
            outputs.append(f"{header}\n{summary_text}")

        if not outputs:
            return "No summaries could be generated."
            
    except Exception as e:
        success = False
        error_msg = str(e)
        return f"Error generating summary: {str(e)}"
    finally:
        duration = time.time() - start_time
        metrics_tracker.log_inference("Stay Summarizer", duration, success, error_msg)

    return "\n\n---\n\n".join(outputs)

def run_manual_agent(stay_data: dict) -> str:
    """
    Summarize a single stay from manual input.
    """
    summarizer = SummarizerML()
    print("--- AGENT: Summarizing manual entry ---")
    
    start_time = time.time()
    success = True
    error_msg = None
    
    try:
        summary_text = summarizer.summarize_stay(stay_data)
        return summary_text
    except Exception as e:
        success = False
        error_msg = str(e)
        return f"Error generating summary: {str(e)}"
    finally:
        duration = time.time() - start_time
        metrics_tracker.log_inference("Stay Summarizer", duration, success, error_msg)
