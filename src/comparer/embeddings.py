"""Embedding providers.

The pipeline talks to embeddings through a tiny interface so the provider
can change without touching retrieval. The default is a LOCAL model
(sentence-transformers ``all-MiniLM-L6-v2``) so the project runs with no
API key and no per-call cost. If that model cannot be loaded (no network,
no torch), we fall back to a deterministic hashing embedder and say so
loudly — retrieval quality drops, but the pipeline stays runnable and the
eval harness keeps reporting honest numbers.

Set ``EMBEDDER=hashing`` to force the fallback, or ``EMBEDDER=local`` to
require the local model. An API embedder can be added later behind the
same interface (``EMBEDDER=api`` is reserved).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    "a an the and or of to in for on at by with will would is are was were be been "
    "it its this that these those as from has have had not no do does did".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (offline fallback).

    Each token is hashed into a fixed-dimension space; the vector is the
    L2-normalised token histogram. Similarity is therefore token-overlap
    based — crude, but real, and fully reproducible in tests and CI.
    The space is deliberately much wider than MiniLM's 384 dims: with a
    histogram this sparse, a narrow space produces hash collisions that
    inflate the apparent similarity of unrelated texts.
    """

    name = "hashing"
    dim = 4096

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbedder:
    """Local embedding model: all-MiniLM-L6-v2 (384-dim, normalised)."""

    name = "sentence-transformers:all-MiniLM-L6-v2"
    dim = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


_embedder_cache = None


def get_embedder():
    """Return the configured embedder, falling back loudly if needed."""
    global _embedder_cache
    if _embedder_cache is not None:
        return _embedder_cache

    choice = os.environ.get("EMBEDDER", "local").lower()
    if choice == "hashing":
        _embedder_cache = HashingEmbedder()
        return _embedder_cache

    try:
        _embedder_cache = SentenceTransformerEmbedder()
    except Exception as exc:  # import failure, download failure, no torch…
        print(
            f"[embeddings] WARNING: local model unavailable ({exc!r}); "
            "falling back to the hashing embedder. Retrieval quality will "
            "be lower and eval numbers reflect the fallback.",
            file=sys.stderr,
        )
        _embedder_cache = HashingEmbedder()
    return _embedder_cache
