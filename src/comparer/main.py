"""FastAPI application for the Toronto Candidate Policy Comparer.

Routes:
* ``GET /`` — the single-page web UI (static/index.html).
* ``GET /health`` — liveness.
* ``POST /ask`` — runs the answering pipeline (classify → per-candidate
  balanced retrieval → generation → critic pass) and returns the
  structured, cited result. Returns 503 until the corpus has been
  ingested (`python -m comparer.ingest`), because an answer with no
  corpus behind it would be exactly the kind of fabrication this
  product exists to prevent.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .answer import CorpusNotReady, get_pipeline

app = FastAPI(title="Toronto Candidate Policy Comparer")

_INDEX = Path(__file__).parent / "static" / "index.html"


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_INDEX, media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: AskRequest):
    try:
        pipeline = get_pipeline()
    except CorpusNotReady as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    return pipeline.answer(payload.question)
