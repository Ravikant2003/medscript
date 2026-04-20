"""Pydantic schemas for MedScript API."""

from __future__ import annotations

from pydantic import BaseModel


class NoteRequest(BaseModel):
    note: str


class SOAPResponse(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class ErrorResponse(BaseModel):
    error: str
    missing_sections: list
