"""Hybrid retrieval with per-candidate balancing.

Design (from the product spec):

* Scores come from two rankers over each candidate's own chunks — cosine
  similarity over embeddings, and BM25 over tokens — fused with Reciprocal
  Rank Fusion (RRF). Neither ranker is trusted alone: embeddings catch
  paraphrase, BM25 catches exact figures and names.
* Retrieval runs **per candidate**: top-k is taken from each candidate's
  chunks separately, then merged. A single global ranking would let the
  candidate with more published material (or longer documents) crowd the
  other out of the context window — balancing is a mechanism here, not
  something we ask the model to do politely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .embeddings import cosine, get_embedder, tokenize
from .schema import Chunk

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused RRF score within the candidate's ranking
    vector_score: float
    bm25_score: float


class _BM25:
    """BM25 over a fixed document set. Uses rank_bm25 when installed,
    otherwise a small built-in implementation with the same API."""

    def __init__(self, tokenized_docs: list[list[str]]) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._impl = BM25Okapi(tokenized_docs)
            self._docs = None
        except Exception:
            self._impl = None
            self._docs = tokenized_docs
            self._build()

    def _build(self) -> None:
        docs = self._docs or []
        self._n = len(docs)
        self._avgdl = (sum(len(d) for d in docs) / self._n) if self._n else 0.0
        self._df: dict[str, int] = {}
        for doc in docs:
            for tok in set(doc):
                self._df[tok] = self._df.get(tok, 0) + 1
        self._tf: list[dict[str, int]] = []
        for doc in docs:
            counts: dict[str, int] = {}
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
            self._tf.append(counts)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        if self._impl is not None:
            return list(map(float, self._impl.get_scores(query_tokens)))
        k1, b = 1.5, 0.75
        scores: list[float] = []
        for i, doc in enumerate(self._docs or []):
            dl = len(doc) or 1
            score = 0.0
            for tok in set(query_tokens):
                df = self._df.get(tok, 0)
                if not df:
                    continue
                idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
                tf = self._tf[i].get(tok, 0)
                denom = tf + k1 * (1 - b + b * dl / (self._avgdl or 1.0))
                score += idf * (tf * (k1 + 1) / denom)
            scores.append(score)
        return scores


@dataclass
class Retriever:
    chunks: list[Chunk]
    embedder: object = field(default=None)
    _by_candidate: dict[str, list[Chunk]] = field(default_factory=dict, init=False)
    _vectors: dict[str, list[float]] = field(default_factory=dict, init=False)
    _bm25: dict[str, _BM25] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = get_embedder()
        for chunk in self.chunks:
            self._by_candidate.setdefault(chunk.candidate, []).append(chunk)
        if self.chunks:
            vectors = self.embedder.embed([c.text for c in self.chunks])
            self._vectors = {c.id: v for c, v in zip(self.chunks, vectors)}
        for candidate, subset in self._by_candidate.items():
            self._bm25[candidate] = _BM25([tokenize(c.text) for c in subset])

    @property
    def candidates(self) -> list[str]:
        return sorted(self._by_candidate)

    def _rank_candidate(self, question: str, candidate: str) -> list[RetrievedChunk]:
        subset = self._by_candidate.get(candidate, [])
        if not subset:
            return []
        q_vec = self.embedder.embed_one(question)
        vec_scores = [cosine(q_vec, self._vectors[c.id]) for c in subset]
        bm25_scores = self._bm25[candidate].get_scores(tokenize(question))

        def ranks(scores: list[float]) -> dict[int, int]:
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            return {idx: rank for rank, idx in enumerate(order)}

        vec_ranks = ranks(vec_scores)
        bm25_ranks = ranks(bm25_scores)
        ranked: list[RetrievedChunk] = []
        for i, chunk in enumerate(subset):
            fused = 1.0 / (RRF_K + vec_ranks[i]) + 1.0 / (RRF_K + bm25_ranks[i])
            ranked.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=fused,
                    vector_score=vec_scores[i],
                    bm25_score=bm25_scores[i],
                )
            )
        ranked.sort(key=lambda r: -r.score)
        return ranked

    def retrieve(
        self, question: str, k_per_candidate: int = 5, candidates: list[str] | None = None
    ) -> dict[str, list[RetrievedChunk]]:
        """Top-k chunks per candidate, keyed by candidate."""
        targets = candidates if candidates is not None else self.candidates
        return {
            candidate: self._rank_candidate(question, candidate)[:k_per_candidate]
            for candidate in targets
        }

    def retrieve_ranked(
        self, question: str, candidate: str
    ) -> list[RetrievedChunk]:
        """Full ranked list for one candidate (used for relevance gating)."""
        return self._rank_candidate(question, candidate)
