"""The critic pass — verification between generation and the reader.

Two checks run on every generated answer:

1. **Structural:** every claim must carry at least one citation, every
   citation must resolve to a real chunk id, and a claim for candidate X
   may not cite a chunk belonging to candidate Y (misattribution — the
   severity-1 defect for this product).
2. **Support:** the embedding similarity between a claim's text and its
   cited chunk(s) must clear a threshold. A claim can carry a perfectly
   real citation that does not actually support it; the citation
   resolving is necessary, not sufficient.

Flagged claims are removed from the answer rather than shown with a
warning — a comparer that prints claims it knows are unsupported has
failed at its only job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .embeddings import cosine, get_embedder
from .generation import GeneratedAnswer
from .schema import Chunk

# MiniLM/hashing cosine between a short claim and a long supporting chunk
# typically lands in 0.25–0.6; below this we treat support as not shown.
SUPPORT_THRESHOLD = 0.20


@dataclass
class CriticReport:
    verdict: str  # "pass" | "fail" (fail = at least one claim removed)
    claims_checked: int = 0
    claims_flagged: int = 0
    flags: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "claims_checked": self.claims_checked,
            "claims_flagged": self.claims_flagged,
            "flags": self.flags,
        }


def verify_answer(
    answer: GeneratedAnswer,
    chunks_by_id: dict[str, Chunk],
    embedder=None,
    support_threshold: float = SUPPORT_THRESHOLD,
) -> tuple[GeneratedAnswer, CriticReport]:
    """Check every claim; return the cleaned answer and the report."""
    if embedder is None:
        embedder = get_embedder()
    report = CriticReport(verdict="pass")

    for candidate, cand_answer in answer.per_candidate.items():
        kept = []
        for claim in cand_answer.claims:
            report.claims_checked += 1
            reasons: list[str] = []
            if not claim.citations:
                reasons.append("no_citation")
            cited = [chunks_by_id[cid] for cid in claim.citations if cid in chunks_by_id]
            if claim.citations and not cited:
                reasons.append("unresolvable_citation")
            for cited_chunk in cited:
                if (
                    cited_chunk.candidate in ("chow", "bradford")
                    and candidate in ("chow", "bradford")
                    and cited_chunk.candidate != candidate
                ):
                    reasons.append("misattributed_citation")
            if cited and not reasons:
                claim_vec = embedder.embed_one(claim.text)
                best = max(cosine(claim_vec, embedder.embed_one(c.text)) for c in cited)
                if best < support_threshold:
                    reasons.append(f"low_support:{best:.2f}<{support_threshold}")
            if reasons:
                report.claims_flagged += 1
                report.flags.append(
                    {"candidate": candidate, "claim": claim.text[:120], "reasons": reasons}
                )
            else:
                kept.append(claim)
        cand_answer.claims = kept
        if not kept and cand_answer.status == "answered":
            cand_answer.status = "not_addressed"

    if report.claims_flagged:
        report.verdict = "fail"
    return answer, report
