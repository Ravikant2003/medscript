"""Inference wrapper for MedScript model (transformers + PEFT, HF Hub adapter)."""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER    = "Ravi2003/medscript-qwen2.5-3b-qlora"

SYSTEM_PROMPT = (
    "You are a clinical assistant. Given an unstructured doctor's note, "
    "generate a structured SOAP summary."
)

model     = None
tokenizer = None


def load_model():
    global model, tokenizer

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print("Using CUDA GPU")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU")

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device,
    )

    print("Loading adapters...")
    model = PeftModel.from_pretrained(base, ADAPTER)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    model.eval()
    print("Model ready!")

def generate_soap(note: str) -> str:
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{note}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def parse_soap(text: str) -> dict:
    sections = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
    patterns = {
        "subjective": r"S[:\s](.*?)(?=O[:\s]|$)",
        "objective":  r"O[:\s](.*?)(?=A[:\s]|$)",
        "assessment": r"A[:\s](.*?)(?=P[:\s]|$)",
        "plan":       r"P[:\s](.*?)$",
    }
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section] = match.group(1).strip()
    return sections
