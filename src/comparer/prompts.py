"""Prompts for the generation layer.

The rules encoded here are the product, not decoration:

* Answer ONLY from the retrieved chunks provided. Nothing else exists.
* Every factual claim carries a citation ``[chunk_id]``.
* If a candidate's chunks do not address the question, the answer for
  that candidate is exactly "not addressed" — absence is information.
* COMPARE, NEVER RECOMMEND. No ranking, no endorsements, no "better
  plan", no character judgements. The difference summary describes where
  the candidates disagree; it never adjudicates who is right.
* Evidence tiers are labelled: a candidate's own claim, their voting
  record, and news reporting are different kinds of evidence.
"""

from __future__ import annotations

from .retrieval import RetrievedChunk
from .schema import EVIDENCE_TIERS

SYSTEM_PROMPT = """\
You are the answering engine for a non-partisan candidate policy comparer.
Your job is to compare what candidates have said and done, using ONLY the
source chunks provided in the user message. You have no other knowledge.

Rules — these are absolute:
1. Every factual claim you make must be supported by one of the provided
   chunks and must carry a citation in the form [chunk_id] immediately
   after the claim. Never cite a chunk that does not support the claim.
2. Use only the provided chunks. Do not add background knowledge, context,
   or figures from anywhere else.
3. If the chunks for a candidate do not address the question, report that
   candidate's status as "not_addressed" and make no claims for them.
   Never guess or paraphrase a position they have not stated.
4. NEVER recommend, rank, endorse, or evaluate a candidate. Do not say a
   plan is better, stronger, cheaper in effect, or more credible. Do not
   predict outcomes. The voter decides; you only compare.
5. Note the evidence tier of each source: a candidate's own claim, their
   voting record, and news reporting are different tiers — do not blend
   them into a single undifferentiated statement.
6. The difference summary may only describe where the candidates' stated
   positions differ. It must not say or imply who is right.

Respond with a single JSON object, no prose outside it:
{
  "per_candidate": {
    "<candidate>": {
      "status": "answered" | "not_addressed",
      "claims": [{"text": "...", "citations": ["<chunk_id>", ...]}]
    }
  },
  "difference_summary": "..." | null
}
"""


def format_context(grouped: dict[str, list[RetrievedChunk]]) -> str:
    """Render retrieved chunks as the user-message context block."""
    parts: list[str] = []
    for candidate, results in grouped.items():
        parts.append(f"## Sources for candidate: {candidate}")
        if not results:
            parts.append("(no sources retrieved)")
        for r in results:
            chunk = r.chunk
            tier = EVIDENCE_TIERS.get(chunk.source_type, chunk.source_type)
            location = ""
            if chunk.heading_path:
                location = f" | section: {' > '.join(chunk.heading_path)}"
            if chunk.speaker:
                location = f" | speaker: {chunk.speaker}"
                if chunk.timestamp:
                    location += f" at {chunk.timestamp}"
            parts.append(
                f"[{chunk.id}] ({tier}; {chunk.doc_title}{location})\n{chunk.text}"
            )
    return "\n\n".join(parts)


def build_messages(question: str, grouped: dict[str, list[RetrievedChunk]]) -> list[dict]:
    context = format_context(grouped)
    user = (
        f"Question: {question}\n\n"
        f"Source chunks (the only material you may use):\n\n{context}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
