"""Tests for the API shell and the /ask wiring.

The pipeline itself is covered by the chunking / retrieval / critic unit
tests; here we verify the HTTP surface. /ask returns 503 until a corpus
has been ingested — the test points the pipeline at a nonexistent chunks
file so it is independent of whatever corpus exists on the dev machine.
"""

import pytest
from fastapi.testclient import TestClient

from comparer import answer
from comparer.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_without_corpus_is_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COMPARER_CHUNKS_PATH", "/nonexistent/chunks.jsonl")
    answer.reset_pipeline()
    resp = client.post("/ask", json={"question": "Where do they differ on property taxes?"})
    assert resp.status_code == 503
    assert "chunks file not found" in resp.json()["detail"]
    answer.reset_pipeline()
