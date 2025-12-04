# ml_model.py
# GenMedX — Seq2Seq inference utilities
# Loads a fine-tuned model from Hugging Face Hub and generates summaries.

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import pandas as pd
except Exception:
    pd = None

# ==============================================================================
# Configuration
# ==============================================================================
# Default to the fine-tuned model on Hugging Face
DEFAULT_MODEL_ID = "kushbindal/genmedx-t5-small"

# Generation settings
MAX_NEW_TOKENS       = int(os.environ.get("MAX_NEW_TOKENS", "160"))
NUM_BEAMS            = int(os.environ.get("NUM_BEAMS", "4"))
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", "4"))
LENGTH_PENALTY       = float(os.environ.get("LENGTH_PENALTY", "1.0"))
REPETITION_PENALTY   = float(os.environ.get("REPETITION_PENALTY", "2.0"))
MIN_NEW_TOKENS       = int(os.environ.get("MIN_NEW_TOKENS", "5"))

# Device selection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ==============================================================================
# Load model/tokenizer (dynamic & cached)
# ==============================================================================
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model_and_tokenizer(model_id: str = DEFAULT_MODEL_ID):
    print(f"🔄 Loading Summarizer Model: {model_id} on {DEVICE}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.to(DEVICE)
        model.eval()

        # Fallback pad token handling
        if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token_id"):
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Summarizer Model Loaded!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model {model_id}: {e}")
        raise e

# ==============================================================================
# Helpers: safe field access & serialization
# ==============================================================================
def _to_str(x: Optional[object]) -> str:
    return "" if x is None else str(x).strip()

def _first_not_null(values: List[object]) -> Optional[str]:
    for v in values:
        s = _to_str(v)
        if s:
            return s
    return None

def _clean_diagnoses(raw_diags: List[object] | object) -> List[str]:
    """Normalize and deduplicate diagnosis strings."""
    if isinstance(raw_diags, list):
        diags = [str(x).strip() for x in raw_diags if str(x).strip()]
    elif raw_diags is None:
        diags = []
    else:
        diags = [str(raw_diags).strip()]
    
    seen = set()
    diags_unique: List[str] = []
    for d in diags:
        if d and d not in seen:
            diags_unique.append(d)
            seen.add(d)
    return diags_unique

def _extract_from_df(stay_df: "pd.DataFrame") -> Dict[str, Optional[str]]:
    """Extract canonical fields from a grouped DataFrame representing one stay."""
    row0 = stay_df.iloc[0]

    raw_diags = []
    if "icd_title" in stay_df.columns:
        raw_diags = stay_df["icd_title"].dropna().astype(str).tolist()
    
    diags_unique = _clean_diagnoses(raw_diags)

    return {
        "intime": _to_str(getattr(row0, "intime", None)),
        "arrival_transport": _to_str(getattr(row0, "arrival_transport", None)),
        "chiefcomplaint": _to_str(getattr(row0, "chiefcomplaint", None)),
        "temperature": _to_str(getattr(row0, "temperature", None)),
        "heartrate": _to_str(getattr(row0, "heartrate", None)),
        "resprate": _to_str(getattr(row0, "resprate", None)),
        "o2sat": _to_str(getattr(row0, "o2sat", None)),
        "sbp": _to_str(getattr(row0, "sbp", None)),
        "dbp": _to_str(getattr(row0, "dbp", None)),
        "disposition": _to_str(getattr(row0, "disposition", None)),
        "diagnoses": diags_unique,
    }

def _extract_from_dict(stay: Dict[str, object]) -> Dict[str, Optional[str]]:
    """Extract canonical fields from a dict using CSV-like keys."""
    raw_diags = stay.get("diagnoses") or stay.get("icd_titles") or stay.get("icd_title")
    diags = _clean_diagnoses(raw_diags)

    return {
        "intime": _to_str(stay.get("intime")),
        "arrival_transport": _to_str(stay.get("arrival_transport")),
        "chiefcomplaint": _to_str(stay.get("chiefcomplaint")),
        "temperature": _to_str(stay.get("temperature")),
        "heartrate": _to_str(stay.get("heartrate")),
        "resprate": _to_str(stay.get("resprate")),
        "o2sat": _to_str(stay.get("o2sat")),
        "sbp": _to_str(stay.get("sbp")),
        "dbp": _to_str(stay.get("dbp")),
        "disposition": _to_str(stay.get("disposition")),
        "diagnoses": diags,
    }

def serialize_stay(stay: Union[Dict[str, object], "pd.DataFrame"]) -> str:
    """
    Build the compact, ordered serialization expected by the fine-tuned model.
    """
    if pd is not None and hasattr(stay, "iloc"):
        fields = _extract_from_df(stay)
    else:
        fields = _extract_from_dict(stay) 

    visit_date = _first_not_null([
        fields.get("intime", "").split(" ")[0] if fields.get("intime") else None
    ]) or ""

    arrival = fields.get("arrival_transport") or ""
    complaint = fields.get("chiefcomplaint") or ""
    temp = fields.get("temperature") or ""
    hr = fields.get("heartrate") or ""
    rr = fields.get("resprate") or ""
    o2 = fields.get("o2sat") or ""
    sbp = fields.get("sbp") or ""
    dbp = fields.get("dbp") or ""
    dispo = fields.get("disposition") or ""
    diagnoses: List[str] = fields.get("diagnoses") or []
    diag_block = "; ".join(diagnoses) if diagnoses else ""

    slots = (
        f"DATE: {visit_date} | "
        f"ARRIVAL: {arrival} | "
        f"COMPLAINT: {complaint} | "
        f"VITALS: T {temp} °F; HR {hr}/min; RR {rr}/min; SpO2 {o2}%; BP {sbp}/{dbp} mmHg | "
        f"DISPOSITION: {dispo} | "
        f"DIAGNOSES: {diag_block}"
    ).strip()

    return "summarize: " + slots

# ==============================================================================
# Generation
# ==============================================================================
@torch.inference_mode()
def generate_summary(serialized_text: str, model_id: str = DEFAULT_MODEL_ID) -> str:
    """Run decoding with safe, concise settings."""
    model, tokenizer = get_model_and_tokenizer(model_id)

    encoded = tokenizer(
        serialized_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    output_ids = model.generate(
        **encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        do_sample=False,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        length_penalty=LENGTH_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
        early_stopping=True
    )
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return _clean_text(text)

def _clean_text(text: str) -> str:
    """Light normalization & de-instruction cleanup."""
    s = text.strip()
    if s.lower().startswith("summarize:"):
        s = s[len("summarize:"):].strip()
    
    s = " ".join(s.split())
    s = s.replace(" %", "%").replace(" .", ".")
    s = s.replace("° F", "°F")
    return s.strip()

class SummarizerML:
    """Wrapper for the summarizer model"""
    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id

    def summarize_stay(self, stay) -> str:
        prompt = serialize_stay(stay)
        print(f"DEBUG: Prompt sent to model:\n{prompt}")
        
        text = generate_summary(prompt, model_id=self.model_id)
        print(f"DEBUG: Raw model output:\n{text}")
        
        # If the model omitted a diagnosis phrase, append it once (polite patch).
        # This logic was present in the original agent code.
        need_dx = ("diagnosis" not in text.lower()) and ("DIAGNOSES:" in prompt)
        if need_dx:
            def _grab(tag: str) -> str:
                start = prompt.find(tag)
                if start == -1:
                    return ""
                start += len(tag)
                end = prompt.find(" | ", start)
                return prompt[start:end if end != -1 else None].strip()

            dx = _grab("DIAGNOSES:")
            if dx:
                text = (text.rstrip(". ") + f". Primary diagnosis: {dx}.").strip()
        
        return text
