"""The critic pass: unsupported and unresolvable claims must be caught."""

from comparer.critic import verify_answer
from comparer.embeddings import HashingEmbedder
from comparer.generation import CandidateAnswer, Claim, GeneratedAnswer
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


def test_unsupported_and_fake_citations_are_flagged():
    lunch = _chunk("c1", "chow", "Chow will provide free lunch in schools to give parents grocery bill relief across Toronto.")
    tax = _chunk("b1", "bradford", "Bradford will freeze property taxes for one year while an independent review examines city spending.")
    chunks_by_id = {"c1": lunch, "b1": tax}

    answer = GeneratedAnswer(
        per_candidate={
            "chow": CandidateAnswer(
                "answered",
                [
                    Claim("Chow will provide free lunch in schools to give parents grocery bill relief.", ["c1"]),
                    # Unsupported: the cited chunk is about school lunch, not taxes.
                    Claim("Chow will freeze property taxes for one year.", ["c1"]),
                    # Unresolvable: no such chunk exists.
                    Claim("Chow promised a new airport on the waterfront.", ["nope-999"]),
                    # Misattributed: a Bradford chunk cited under Chow.
                    Claim("Property taxes will be frozen for one year during a spending review.", ["b1"]),
                ],
            ),
            "bradford": CandidateAnswer("not_addressed", []),
        },
        difference_summary=None,
    )

    cleaned, report = verify_answer(answer, chunks_by_id, embedder=HashingEmbedder())
    assert report.verdict == "fail"
    assert report.claims_checked == 4
    assert report.claims_flagged == 3
    kept = cleaned.per_candidate["chow"].claims
    assert len(kept) == 1
    assert "lunch" in kept[0].text
    reasons = {tuple(f["reasons"]) for f in report.flags}
    assert any("unresolvable_citation" in r for r in reasons)
    assert any("misattributed_citation" in r for r in reasons)
