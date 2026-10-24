"""Per-candidate balanced retrieval.

Fixture: Chow has 20 chunks about housing, Bradford has 2. A single
global ranking would fill the context with Chow; per-candidate retrieval
must still surface Bradford's material at the requested k.
"""

from comparer.embeddings import HashingEmbedder
from comparer.retrieval import Retriever
from comparer.schema import Chunk


def _chunk(cid: str, candidate: str, text: str) -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        candidate=candidate,
        source_type="platform",
        source_url="https://example.ca",
        fetched_date="2026-10-23",
        doc_title="Doc",
        chunk_index=0,
    )


def _corpus() -> list[Chunk]:
    chunks = [
        _chunk(f"chow-{i}", "chow", f"Chow housing plan detail {i}: build rental housing near transit, housing housing housing.")
        for i in range(20)
    ]
    chunks += [
        _chunk("bradford-0", "bradford", "Bradford housing plan: build more homes faster, cut development charges on housing."),
        _chunk("bradford-1", "bradford", "Bradford will approve housing projects within thirty days of application."),
    ]
    return chunks


def test_minority_candidate_is_represented():
    retriever = Retriever(_corpus(), embedder=HashingEmbedder())
    grouped = retriever.retrieve("What is the housing plan?", k_per_candidate=2)
    assert len(grouped["chow"]) == 2
    assert len(grouped["bradford"]) == 2
    assert {r.chunk.id for r in grouped["bradford"]} == {"bradford-0", "bradford-1"}
    # Rankings are sorted by fused score, descending.
    for results in grouped.values():
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
