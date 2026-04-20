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
    "generate a structured SOAP summary with clearly labeled sections: "
    "S (Subjective), O (Objective), A (Assessment), P (Plan)."
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
            repetition_penalty=1.15,  # Penalise repeated tokens
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def _clean_section(text: str) -> str:
    """Remove artifacts, deduplicate sentences, and normalise whitespace."""
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leftover special tokens
    text = re.sub(r"<\|.*?\|>", "", text)
    # Remove leading/trailing whitespace on each line
    lines = [ln.strip() for ln in text.splitlines()]
    # Deduplicate consecutive identical lines
    deduped: list[str] = []
    for line in lines:
        if not deduped or line.lower() != deduped[-1].lower():
            deduped.append(line)
    result = " ".join(ln for ln in deduped if ln)
    # Collapse internal multiple spaces
    result = re.sub(r" {2,}", " ", result)
    return result.strip()


def parse_soap(text: str) -> dict:
    """Extract SOAP sections with robust multi-strategy matching."""
    sections: dict[str, str] = {
        "subjective": "",
        "objective": "",
        "assessment": "",
        "plan": "",
    }

    # Strategy 1 — labelled headers like "Subjective:", "S:", "S -"
    labelled = {
        "subjective": r"(?:subjective|S)[:\-\s]\s*(.*?)(?=(?:objective|O)[:\-\s]|$)",
        "objective":  r"(?:objective|O)[:\-\s]\s*(.*?)(?=(?:assessment|A)[:\-\s]|$)",
        "assessment": r"(?:assessment|A)[:\-\s]\s*(.*?)(?=(?:plan|P)[:\-\s]|$)",
        "plan":       r"(?:plan|P)[:\-\s]\s*(.*?)$",
    }
    for section, pattern in labelled.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section] = _clean_section(match.group(1))

    # Strategy 2 — fallback: split on SOAP bullet markers if labelled pass failed
    missing = [k for k, v in sections.items() if not v]
    if missing:
        # Try JSON-style key extraction  e.g. "subjective": "..."
        for section in missing:
            json_match = re.search(
                rf'"{section}"\s*:\s*"(.*?)"(?=\s*[,}}])',
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if json_match:
                sections[section] = _clean_section(json_match.group(1))

    return sections
