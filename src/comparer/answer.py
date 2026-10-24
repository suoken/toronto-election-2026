"""The answering pipeline: classify → plan → retrieve → generate → verify.

Question handling:

* **Refusal** — recommendation-seeking ("who should I vote for"), outcome
  predictions, candidates' private lives, and other jurisdictions are
  refused outright. The comparer compares stated positions and records;
  it never recommends, and it does not answer from outside the corpus.
* **Fact** — a question naming one candidate is answered for that
  candidate; the other is reported ``not_addressed`` (the question was
  not about them).
* **Comparison / difference** — both candidates are retrieved for, with
  per-candidate balancing, and a candidate whose material genuinely does
  not cover the question is reported ``not_addressed`` — a first-class
  outcome, not an error.

Per-candidate relevance gating happens before generation: if a
candidate's best retrieved chunk is not actually similar to the question,
their chunk list is emptied so the generator cannot manufacture a
position from irrelevant text. Similarity alone is not enough — a
candidate can write a lot about *taxes* without ever mentioning the
*land transfer tax* — so the gate also requires coverage: at least one
of the question's distinctive terms (terms that appear in the
candidate's chunks but are not ubiquitous there) must show up in the
retrieved material. Topical adjacency is not coverage.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .critic import verify_answer
from .embeddings import get_embedder, tokenize
from .generation import get_generator
from .retrieval import Retriever
from .schema import Chunk

CANDIDATE_NAMES = {"chow": "Olivia Chow", "bradford": "Brad Bradford"}
KNOWN_CANDIDATES = sorted(CANDIDATE_NAMES)

RELEVANCE_THRESHOLD = float(os.environ.get("COMPARER_RELEVANCE_THRESHOLD", "0.15"))
# Minimum IDF-weighted share of the question's vocabulary that a
# candidate's retrieved chunks must cover to count as "addressed".
COVERAGE_MIN_RATIO = float(os.environ.get("COMPARER_COVERAGE_MIN_RATIO", "0.5"))

_REFUSAL_PATTERNS: list[tuple[str, str]] = [
    (r"\bwho (should|shall) i vote", "The comparer does not recommend a candidate."),
    (r"\bwho is the better (candidate|mayor|choice)", "The comparer does not rank or evaluate candidates."),
    (r"\b(recommend|suggest) (a|the|who)\b.*\b(vote|candidate|mayor)", "The comparer does not recommend a candidate."),
    (r"\bwho (will|is going to) win\b", "The comparer does not predict election outcomes."),
    (r"\b(poll|polls|polling)\b", "The comparer answers from platforms, debates and records — polling is out of scope."),
    (r"\b(home address|where does [\w ]{0,30}? live|wife|husband|spouse|religion|ethnicity|height|salary|net worth)\b",
     "The comparer covers stated positions and public records, not candidates' private lives."),
    (r"\b(chow|bradford)('s|’s) (family|wife|husband|kids|children)\b",
     "The comparer covers stated positions and public records, not candidates' private lives."),
    (r"\bweather\b", "Out of scope for a candidate policy comparer."),
    (r"\b(mayor of (vancouver|ottawa|montreal|calgary|edmonton|mississauga|brampton))\b",
     "The comparer covers the Toronto mayoral race only."),
]


def classify_question(question: str) -> tuple[str, str | None]:
    """Return (type, refusal_reason). Type is refusal|fact|comparison|difference."""
    q = question.lower()
    for pattern, reason in _REFUSAL_PATTERNS:
        if re.search(pattern, q):
            return "refusal", reason
    named = [c for c in KNOWN_CANDIDATES if c in q]
    if re.search(r"\b(differ|difference|compare|comparison|versus|vs\.?|both)\b", q):
        return "difference", None
    if len(named) == 1:
        return "fact", None
    return "comparison", None


class CorpusNotReady(RuntimeError):
    pass


class Pipeline:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder=None,
        generator=None,
        k_per_candidate: int = 6,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.generator = generator or get_generator()
        self.retriever = Retriever(chunks, embedder=self.embedder)
        self.chunks_by_id = {c.id: c for c in chunks}
        self.k_per_candidate = k_per_candidate
        self.relevance_threshold = relevance_threshold
        # Document frequency per candidate, for the coverage gate.
        self._doc_freq: dict[str, dict[str, int]] = {}
        self._candidate_sizes: dict[str, int] = {}
        self._corpus_df: dict[str, int] = {}
        self._total_chunks = len(chunks)
        for chunk in chunks:
            df = self._doc_freq.setdefault(chunk.candidate, {})
            self._candidate_sizes[chunk.candidate] = (
                self._candidate_sizes.get(chunk.candidate, 0) + 1
            )
            for term in set(tokenize(chunk.text)):
                df[term] = df.get(term, 0) + 1
                self._corpus_df[term] = self._corpus_df.get(term, 0) + 1

    def _coverage_ok(self, candidate: str, question: str, top_texts: list[str]) -> bool:
        """True when the candidate's retrieved material covers enough of
        the question's *distinctive* vocabulary.

        Each question term is weighted by its corpus-wide IDF (rare terms
        carry the mass), and we ask what share of that mass appears in the
        candidate's top retrieved chunks. Missing a rare term — "marathon"
        in a question about the marathon pledge, asked of the candidate
        who never mentioned it — cannot be compensated by generic shared
        vocabulary like "subway" or "tax". Terms unknown to the whole
        corpus carry no evidence and are ignored. If no term carries any
        mass, fall back to the similarity gate alone."""
        import math

        present: set[str] = set()
        for text in top_texts:
            present |= set(tokenize(text))
        covered_mass = 0.0
        total_mass = 0.0
        for term in set(tokenize(question)):
            df_all = self._corpus_df.get(term, 0)
            if df_all == 0:
                continue
            idf = math.log(1 + self._total_chunks / df_all)
            total_mass += idf
            if term in present:
                covered_mass += idf
        if total_mass == 0:
            return True
        return (covered_mass / total_mass) >= COVERAGE_MIN_RATIO

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs) -> "Pipeline":
        path = Path(path)
        if not path.exists():
            raise CorpusNotReady(f"chunks file not found: {path}")
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not chunks:
            raise CorpusNotReady(f"chunks file is empty: {path}")
        return cls(chunks, **kwargs)

    def answer(self, question: str) -> dict:
        qtype, refusal_reason = classify_question(question)
        base = {
            "question": question,
            "type": qtype,
            "refused": False,
            "refusal_reason": None,
            "per_candidate": {
                c: {"status": "not_addressed", "claims": []} for c in KNOWN_CANDIDATES
            },
            "difference_summary": None,
            "candidate_names": CANDIDATE_NAMES,
        }
        if qtype == "refusal":
            base["refused"] = True
            base["refusal_reason"] = refusal_reason
            base["trace"] = {"chunks_retrieved": {}, "critic": None}
            return base

        named = [c for c in KNOWN_CANDIDATES if c in question.lower()]
        scoped = named[0] if (qtype == "fact" and len(named) == 1) else None

        grouped = {}
        retrieved_ids: dict[str, list[str]] = {}
        for candidate in KNOWN_CANDIDATES:
            if scoped is not None and candidate != scoped:
                grouped[candidate] = []
                retrieved_ids[candidate] = []
                continue
            ranked = self.retriever.retrieve_ranked(question, candidate)
            top_vec = max((r.vector_score for r in ranked[:10]), default=0.0)
            top = ranked[: self.k_per_candidate]
            covered = self._coverage_ok(candidate, question, [r.chunk.text for r in top])
            if top_vec < self.relevance_threshold or not covered:
                grouped[candidate] = []
                retrieved_ids[candidate] = []
            else:
                grouped[candidate] = top
                retrieved_ids[candidate] = [r.chunk.id for r in top]

        generated = self.generator.generate(question, grouped)
        generated, report = verify_answer(generated, self.chunks_by_id, self.embedder)

        base["per_candidate"] = {
            c: generated.per_candidate[c].to_dict() for c in KNOWN_CANDIDATES
        }
        base["difference_summary"] = generated.difference_summary
        base["trace"] = {
            "chunks_retrieved": retrieved_ids,
            "critic": report.to_dict(),
            "embedder": getattr(self.embedder, "name", "unknown"),
            "generator": getattr(self.generator, "name", "unknown"),
            "scoped_to_candidate": scoped,
        }
        return base


_pipeline: Pipeline | None = None


def chunks_path() -> Path:
    env = os.environ.get("COMPARER_CHUNKS_PATH")
    if env:
        return Path(env)
    from .config import REPO_ROOT

    return REPO_ROOT / "data" / "chunks.jsonl"


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_jsonl(chunks_path())
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None
