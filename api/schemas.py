"""Pydantic schemas for MedScript API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class NoteRequest(BaseModel):
    note: str


class SOAPResponse(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    raw_output: str


class ErrorResponse(BaseModel):
    error: str
    missing_sections: list
