"""SOAP parsing and validation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

SECTION_PATTERNS = {
    "subjective": r"\*\*\s*Subjective\s*:\s*\*\*|\bSubjective\s*:",
    "objective": r"\*\*\s*Objective\s*:\s*\*\*|\bObjective\s*:",
    "assessment": r"\*\*\s*Assessment\s*:\s*\*\*|\bAssessment\s*:",
    "plan": r"\*\*\s*Plan\s*:\s*\*\*|\bPlan\s*:",
}

ORDER = ["subjective", "objective", "assessment", "plan"]


@dataclass
class SOAPParseResult:
    parsed: dict[str, str]
    missing_sections: list[str]


def _find_label_positions(text: str) -> list[tuple[str, int, int]]:
    positions: list[tuple[str, int, int]] = []
    for key, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            positions.append((key, match.start(), match.end()))
    return sorted(positions, key=lambda x: x[1])


def parse_soap_sections(text: str) -> SOAPParseResult:
    """Extract SOAP sections from model text.

    Returns parsed sections and list of missing sections.
    """
    raw = text.strip()
    positions = _find_label_positions(raw)

    parsed = {key: "" for key in ORDER}

    if not positions:
        return SOAPParseResult(parsed=parsed, missing_sections=ORDER.copy())

    for idx, (section, _start, end_label) in enumerate(positions):
        next_start = positions[idx + 1][1] if idx + 1 < len(positions) else len(raw)
        value = raw[end_label:next_start].strip(" \n:-*")
        parsed[section] = value.strip()

    missing = [key for key in ORDER if not parsed[key]]
    return SOAPParseResult(parsed=parsed, missing_sections=missing)


def make_soap_markdown(subjective: str, objective: str, assessment: str, plan: str) -> str:
    return (
        f"**Subjective:** {subjective}\n"
        f"**Objective:** {objective}\n"
        f"**Assessment:** {assessment}\n"
        f"**Plan:** {plan}"
    )
