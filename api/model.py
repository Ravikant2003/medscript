"""Inference wrapper for MedScript model (transformers + PEFT, HF Hub adapter)."""

from __future__ import annotations

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER    = "Ravi2003/medscript-qwen2.5-3b-qlora"

# Strict JSON-first prompt: forces structure and prevents cross-section leakage
SYSTEM_PROMPT = """\
You are a clinical documentation assistant. Given an unstructured doctor's note, \
return ONLY a valid JSON object with exactly these 4 fields:

{
  "subjective": "What the patient reports — symptoms, history, complaints.",
  "objective": "Measurable clinical findings — vitals, labs, ECG, exam.",
  "assessment": "Diagnosis or clinical impression based on S and O.",
  "plan": "Treatment, medications, referrals, and follow-up steps."
}

Rules:
- Do NOT repeat information across sections.
- Do NOT include labels like "S:", "O:", "A:", "P:" inside section values.
- Do NOT add any text outside the JSON object.
- Keep each section concise, complete, and clinically accurate.
- If a field has no relevant information, use "Not documented."
"""

model     = None
tokenizer = None


def load_model() -> None:
    global model, tokenizer

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
        print("Using CUDA GPU")
    else:
        device, dtype = "cpu", torch.float32
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
    """Run inference and return the raw assistant output string."""
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
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

_SOAP_LABEL_RE = re.compile(
    r"^\s*(?:subjective|objective|assessment|plan|[SOAP])\s*[:.\-]\s*",
    re.IGNORECASE,
)

_INCOMPLETE_SENTENCE_RE = re.compile(r"[^.!?]\s*$")


def _clean_section(text: str) -> str:
    """
    Normalise a single SOAP section value:
    1. Strip leftover special tokens / inline SOAP labels.
    2. Deduplicate consecutive identical sentences.
    3. Remove an incomplete trailing sentence (no closing punctuation).
    4. Collapse excess whitespace.
    """
    # Remove special tokens
    text = re.sub(r"<\|.*?\|>", "", text)
    # Remove inline SOAP labels at the start of lines (e.g. "O: ...", "Plan:")
    lines = [_SOAP_LABEL_RE.sub("", ln).strip() for ln in text.splitlines()]
    # Drop empty lines and deduplicate consecutive identical ones
    deduped: list[str] = []
    for line in lines:
        if line and (not deduped or line.lower() != deduped[-1].lower()):
            deduped.append(line)

    result = " ".join(deduped)
    result = re.sub(r" {2,}", " ", result).strip()

    # Trim an incomplete trailing sentence (no terminal punctuation)
    if result and _INCOMPLETE_SENTENCE_RE.search(result):
        # Find the last sentence boundary and cut there
        last_boundary = max(result.rfind("."), result.rfind("!"), result.rfind("?"))
        if last_boundary > len(result) // 2:  # Only trim if meaningful content remains
            result = result[: last_boundary + 1].strip()

    return result or "Not documented."


def parse_soap(raw: str) -> dict[str, str]:
    """
    Extract SOAP sections from raw model output.

    Strategy order (most → least reliable):
      1. JSON parsing  — model followed instructions perfectly
      2. Regex on named headers — e.g. "Subjective:", "S:"
      3. Fallback      — "Not documented." for any missing section
    """
    FIELDS = ("subjective", "objective", "assessment", "plan")
    sections: dict[str, str] = {f: "" for f in FIELDS}

    # ── Strategy 1: JSON ────────────────────────────────────────────────────
    # Extract the first {...} block from the output (model may add preamble)
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            for field in FIELDS:
                value = parsed.get(field, "")
                if isinstance(value, str) and value.strip():
                    sections[field] = _clean_section(value)
        except json.JSONDecodeError:
            pass  # Fall through to regex strategies

    # ── Strategy 2: Named-header regex ──────────────────────────────────────
    # Handles outputs like "Subjective: ...\nObjective: ..."
    header_patterns = {
        "subjective": r"(?:subjective|S)\s*[:\-]\s*(.*?)(?=(?:objective|O)\s*[:\-]|$)",
        "objective":  r"(?:objective|O)\s*[:\-]\s*(.*?)(?=(?:assessment|A)\s*[:\-]|$)",
        "assessment": r"(?:assessment|A)\s*[:\-]\s*(.*?)(?=(?:plan|P)\s*[:\-]|$)",
        "plan":       r"(?:plan|P)\s*[:\-]\s*(.*?)$",
    }
    for field in FIELDS:
        if not sections[field]:  # Only fill gaps not resolved by JSON
            m = re.search(header_patterns[field], raw, re.DOTALL | re.IGNORECASE)
            if m:
                sections[field] = _clean_section(m.group(1))

    # ── Strategy 3: Fallback ────────────────────────────────────────────────
    for field in FIELDS:
        if not sections[field]:
            sections[field] = "Not documented."

    return sections
