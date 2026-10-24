"""Chunk persistence (SQLAlchemy).

Mirrors :class:`comparer.schema.Chunk`. Defaults to a local SQLite file so
the project runs with zero infrastructure; set ``DATABASE_URL`` to point
at the Postgres service from docker-compose.yml in production.

Embeddings are stored as a JSON array column. That is a pragmatic
abstraction for tonight: SQLite has no vector type, and the corpus is
small enough (~1–2k chunks) that in-memory cosine over JSON-decoded
vectors is fine. On Postgres the same column can migrate to pgvector
(``ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(384)``) without
changing the rest of the code — TODO when corpus size demands it.

CLI:
    python -m comparer.store load [--jsonl data/chunks.jsonl] [--embed]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sqlalchemy import JSON, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .schema import Chunk

DEFAULT_DB_URL = "sqlite:///data/comparer.db"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


class Base(DeclarativeBase):
    pass


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    candidate: Mapped[str] = mapped_column(String, index=True)
    source_type: Mapped[str] = mapped_column(String, index=True)
    source_url: Mapped[str] = mapped_column(String)
    fetched_date: Mapped[str] = mapped_column(String)
    doc_title: Mapped[str] = mapped_column(String)
    chunk_index: Mapped[int] = mapped_column(Integer)
    char_count: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[list] = mapped_column(JSON, default=list)
    speaker: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[str | None] = mapped_column(String, nullable=True)
    # Embedding vector as a JSON array of floats (see module docstring).
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)

    def to_chunk(self) -> Chunk:
        return Chunk(
            id=self.id,
            text=self.text,
            candidate=self.candidate,
            source_type=self.source_type,
            source_url=self.source_url,
            fetched_date=self.fetched_date,
            doc_title=self.doc_title,
            chunk_index=self.chunk_index,
            char_count=self.char_count,
            heading_path=self.heading_path or [],
            speaker=self.speaker,
            timestamp=self.timestamp,
        )


def get_engine(url: str | None = None):
    url = url or database_url()
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine


def load_chunks_jsonl(engine, jsonl_path: str | Path) -> int:
    """Replace the chunks table with the contents of a chunks.jsonl file."""
    chunks = [
        Chunk.from_dict(json.loads(line))
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with Session(engine) as session:
        session.query(ChunkRow).delete()
        session.add_all(
            ChunkRow(
                id=c.id,
                text=c.text,
                candidate=c.candidate,
                source_type=c.source_type,
                source_url=c.source_url,
                fetched_date=c.fetched_date,
                doc_title=c.doc_title,
                chunk_index=c.chunk_index,
                char_count=c.char_count,
                heading_path=c.heading_path,
                speaker=c.speaker,
                timestamp=c.timestamp,
            )
            for c in chunks
        )
        session.commit()
    return len(chunks)


def fetch_chunks(engine) -> list[Chunk]:
    with Session(engine) as session:
        return [row.to_chunk() for row in session.query(ChunkRow).all()]


def fetch_embeddings(engine) -> dict[str, list[float]]:
    with Session(engine) as session:
        rows = session.query(ChunkRow.id, ChunkRow.embedding).all()
    return {row.id: row.embedding for row in rows if row.embedding}


def embed_missing(engine, embedder) -> int:
    """Compute and store embeddings for chunks that don't have one yet."""
    with Session(engine) as session:
        rows = session.query(ChunkRow).filter(ChunkRow.embedding.is_(None)).all()
        if not rows:
            return 0
        vectors = embedder.embed([row.text for row in rows])
        for row, vec in zip(rows, vectors):
            row.embedding = vec
        session.commit()
        return len(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="comparer.store")
    sub = parser.add_subparsers(dest="command", required=True)
    load = sub.add_parser("load", help="load chunks.jsonl into the database")
    load.add_argument("--jsonl", default="data/chunks.jsonl")
    load.add_argument("--embed", action="store_true", help="also embed chunks missing vectors")
    args = parser.parse_args(argv)

    engine = get_engine()
    if args.command == "load":
        count = load_chunks_jsonl(engine, args.jsonl)
        print(f"loaded {count} chunks from {args.jsonl} into {database_url()}")
        if args.embed:
            from .embeddings import get_embedder

            embedder = get_embedder()
            n = embed_missing(engine, embedder)
            print(f"embedded {n} chunks with {embedder.name}")


if __name__ == "__main__":
    main()
