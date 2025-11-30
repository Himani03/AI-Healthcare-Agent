---
title: Medical Diagnosis System
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Medical Diagnosis System

Two-phase AI pipeline for medical diagnosis and explanation generation.

## Features

- **Phase 1:** Fine-tuned BioMistral-7B classification (99.1% accuracy)
- **Phase 2:** GPT-4o explanation generation with Few-Shot prompting
- **10 Diagnoses:** Infectious gastroenteritis, acute bronchitis, pneumonia, strep throat, anxiety, conjunctivitis due to allergy, eczema, psoriasis, spondylosis, sprain or strain

## Usage

1. Enter symptoms in the specified format
2. Click "Diagnose" to get prediction and explanation
3. View example symptoms for each diagnosis

## Setup

This Space requires two secrets:
- `HF_TOKEN`: Hugging Face token for model access
- `OPENAI_API_KEY`: OpenAI API key for GPT-4o

## Disclaimer

For educational purposes only. Not a substitute for professional medical advice.