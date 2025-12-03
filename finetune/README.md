# 🧠 Fine-Tuning BioMistral-7B

This directory contains scripts to fine-tune the **BioMistral-7B** model on your medical Q&A dataset (`MedQA` + `PubMedQA`) using **QLoRA** (Quantized Low-Rank Adaptation).

## 📋 Prerequisites

Fine-tuning a 7B parameter model requires a GPU with at least **15GB VRAM** (e.g., NVIDIA T4, A10G, A100).
**We highly recommend running this on Google Colab (Free Tier with T4 GPU) or RunPod.**

## 🚀 How to Run on Google Colab (Recommended)

Since we have already generated the dataset, you can run the training directly on Colab without any setup issues.

### Step 1: Upload Files
Create a new notebook in Google Colab and upload these **3 files** to the default content folder:
1.  `train.py`
2.  `requirements.txt`
3.  `medical_chat_dataset.jsonl` (This is your pre-processed data)

### Step 2: Run Training
Copy and paste this code block into a Colab cell and run it. It will install dependencies and start training immediately.

```python
# 1. Install Dependencies
!pip install -r requirements.txt

# 2. Login to Hugging Face (Optional, only if pushing model)
# from huggingface_hub import login
# login()

# 3. Start Training
!python train.py
```

### Step 3: Download Adapter
Once training is complete, the model will be saved in the `BioMistral-7B-GenMedX-Chatbot` folder.
Zip and download this folder to use it in your app!

```python
!zip -r biomistral_adapter.zip BioMistral-7B-GenMedX-Chatbot
from google.colab import files
files.download('biomistral_adapter.zip')
```

---

## 💻 Local Execution (Linux/WSL Only)
*Note: Requires NVIDIA GPU with 15GB+ VRAM.*

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py
```
