"""FastAPI application for the Toronto Candidate Policy Comparer.

Routes:
* ``GET /health`` — liveness.
* ``POST /ask`` — runs the answering pipeline (classify → per-candidate
  balanced retrieval → generation → critic pass) and returns the
  structured, cited result. Returns 503 until the corpus has been
  ingested (`python -m comparer.ingest`), because an answer with no
  corpus behind it would be exactly the kind of fabrication this
  product exists to prevent.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .answer import CorpusNotReady, get_pipeline

app = FastAPI(title="Toronto Candidate Policy Comparer")


class AskRequest(BaseModel):
    question: str


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
