"""Formatting helpers for instruction-tuning chat samples."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a clinical assistant. Given an unstructured doctor's note, "
    "generate a structured SOAP summary."
)


def build_qwen_chat_example(raw_note: str, soap_answer: str) -> str:
    """Build one supervised training text sample with Qwen chat tags."""
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{raw_note}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{soap_answer}<|im_end|>"
    )
