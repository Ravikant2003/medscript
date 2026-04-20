"""FastAPI application for MedScript inference."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import NoteRequest, SOAPResponse, ErrorResponse
from api import model as model_module


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_module.load_model()
    yield


app = FastAPI(
    title="MedScript API",
    description="Clinical note to SOAP summarization via fine-tuned Qwen2.5-3B",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model": "Ravi2003/medscript-qwen2.5-3b-qlora"}


@app.post("/summarize", response_model=SOAPResponse)
def summarize(request: NoteRequest):
    if len(request.note.split()) < 20:
        raise HTTPException(status_code=400, detail="Note too short, minimum 20 words")
    if len(request.note.split()) > 600:
        raise HTTPException(status_code=400, detail="Note too long, maximum 600 words")

    raw = model_module.generate_soap(request.note)
    sections = model_module.parse_soap(raw)

    missing = [k for k, v in sections.items() if not v]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Model output missing sections: {missing}",
        )

    return SOAPResponse(**sections)
