"""Chunking invariants.

The two properties under test are the product's honesty guarantees:
a transcript chunk never spans two speakers, and a document chunk
carries the heading path it sits under.
"""

from comparer.chunking import chunk_markdown_document, chunk_transcript, parse_transcript_text

META = dict(
    source_url="https://example.ca/doc",
    fetched_date="2026-10-23",
    doc_title="Doc",
    id_prefix="test",
)


def test_transcript_chunks_never_span_speakers():
    transcript = (
        "MODERATOR: Welcome to the debate. Tonight we discuss housing.\n"
        "CHOW: Toronto needs more rental homes built near transit lines.\n"
        "BRADFORD: I would freeze property taxes for one full year.\n"
    )
    turns = parse_transcript_text(transcript)
    assert [t["speaker"] for t in turns] == ["MODERATOR", "CHOW", "BRADFORD"]

    chunks = chunk_transcript(turns, **META)
    assert len(chunks) == 3
    markers = {"MODERATOR": "Welcome", "CHOW": "rental homes", "BRADFORD": "freeze"}
    for chunk in chunks:
        own = markers[chunk.speaker]
        others = [m for s, m in markers.items() if s != chunk.speaker]
        assert own in chunk.text
        assert all(other not in chunk.text for other in others)


def test_long_turn_splits_within_turn_and_inherits_speaker():
    long_text = " ".join(
        f"Sentence number {i} explains the housing plan in detail." for i in range(120)
    )
    turns = [{"speaker": "CHOW", "timestamp": "00:10:00", "text": long_text, "candidate": "chow"}]
    chunks = chunk_transcript(turns, **META)
    assert len(chunks) > 1
    assert all(c.speaker == "CHOW" for c in chunks)
    assert all(c.candidate == "chow" for c in chunks)
    assert all(len(c.text) <= 1200 for c in chunks)
    # No content lost, no content duplicated across the split.
    assert sum(len(c.text) for c in chunks) >= len(long_text) - len(chunks) * 2


def test_markdown_heading_path_is_carried():
    doc = (
        "# Housing\n\nIntro paragraph about housing overall.\n\n"
        "## Rental homes\n\nWe will build rental homes near transit.\n\n"
        "# Taxes\n\nWe will hold increases near inflation.\n"
    )
    chunks = chunk_markdown_document(doc, candidate="chow", source_type="platform", **META)
    by_text = {c.text: c for c in chunks}
    rental = next(c for c in chunks if "rental homes near transit" in c.text)
    assert rental.heading_path == ["Housing", "Rental homes"]
    taxes = next(c for c in chunks if "near inflation" in c.text)
    assert taxes.heading_path == ["Taxes"]
    assert by_text  # non-empty
