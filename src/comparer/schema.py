"""Core data schema for the comparer corpus.

A :class:`Chunk` is the atomic unit of evidence. Everything downstream —
retrieval, generation, the critic pass — works with chunks and cites them
by ``id``. The metadata carried on each chunk is what makes per-candidate
balanced retrieval and evidence-tier labelling possible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

SOURCE_TYPES = ("platform", "debate", "record", "news")

# How each source type is presented to the reader. A candidate's own words
# (platform, debate) are *claims*; council votes are their *record*; press
# coverage is *reporting*. The tiers are never blended in the UI.
EVIDENCE_TIERS = {
    "platform": "candidate claim",
    "debate": "candidate claim",
    "record": "voting record",
    "news": "news reporting",
}


@dataclass
class Chunk:
    id: str
    text: str
    candidate: str  # "chow" | "bradford" | "moderator" | "unknown" (debate)
    source_type: str  # one of SOURCE_TYPES
    source_url: str
    fetched_date: str  # ISO date the source was pulled into the corpus
    doc_title: str
    chunk_index: int  # position within the source document
    char_count: int = 0
    # Platforms: the heading hierarchy the chunk sits under, e.g.
    # ["Housing", "Rental homes"]. Empty when the document has no headings.
    heading_path: list[str] = field(default_factory=list)
    # Debates: who is speaking and when (video timestamp if known).
    speaker: str | None = None
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)

    @property
    def evidence_tier(self) -> str:
        return EVIDENCE_TIERS.get(self.source_type, self.source_type)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)
