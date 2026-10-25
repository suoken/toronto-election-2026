"""Corpus ingestion: source files -> chunks.jsonl.

Reads:

* ``corpus/platforms/*.md`` — campaign platform / announcement pages.
  Each file starts with a small header block::

      Source: https://www.oliviachow.ca/plan
      Fetched: 2026-10-23
      Title: Olivia's Plan - Olivia Chow for Mayor of Toronto

  (Bradford files add ``Published:`` and ``Issue category:`` lines.)
  The candidate is taken from the filename prefix (``chow-`` /
  ``bradford-``). Bodies are chunked structure-first by
  :func:`comparer.chunking.chunk_markdown_document`.

* ``corpus/debates/`` — debate transcripts: ``*transcript*.txt`` files
  (parsed for ``Name:`` speaker turns) and ``*.vtt`` caption files
  (YouTube captions; speaker changes are marked but speakers are not
  identified, so those turns are attributed to ``UNKNOWN`` and their
  chunks carry candidate ``"unknown"`` — they are indexed and searchable
  but never attributed to a candidate the transcript does not name).

* ``corpus/records/`` — the Toronto Open Data council voting record.
  The unsplit CSV is read when present; otherwise the four split parts
  (``*-part1.csv`` … ``*-part4.csv``, each with a repeated header) are
  read in order. Only the two mayoral candidates' rows are chunked —
  one chunk per member per vote-event, stating that member's own vote
  (see :func:`comparer.chunking.chunk_voting_records`).

Writes ``data/chunks.jsonl`` (one JSON object per chunk).

    python -m comparer.ingest [--corpus-dir corpus] [--out data/chunks.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from .chunking import (
    chunk_markdown_document,
    chunk_transcript,
    chunk_voting_records,
    parse_transcript_text,
    parse_vtt,
)
from .schema import Chunk

_HEADER_KEYS = ("source", "source url", "fetched", "title", "published", "issue category")
_HEADER_RE = re.compile(r"^([A-Za-z ]{2,20}):\s*(.+)$")


def parse_header(text: str) -> tuple[dict[str, str], str]:
    """Split a platform file into (header dict, body)."""
    header: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            break
        match = _HEADER_RE.match(line)
        if match and match.group(1).strip().lower() in _HEADER_KEYS:
            header[match.group(1).strip().lower()] = match.group(2).strip()
            i += 1
        else:
            break
    body = "\n".join(lines[i:])
    # Drop site-navigation residue left by the page scrape.
    body = "\n".join(ln for ln in body.splitlines() if ln.strip() not in {"Read more", "Read more —", "Read more -"})
    return header, body


def candidate_for_speaker(speaker: str) -> str:
    s = speaker.lower()
    if "chow" in s:
        return "chow"
    if "bradford" in s:
        return "bradford"
    if any(k in s for k in ("paikin", "moderator", "host")):
        return "moderator"
    return "unknown"


def ingest_platforms(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    platform_dir = corpus_dir / "platforms"
    if not platform_dir.exists():
        return chunks
    for path in sorted(platform_dir.glob("*.md")):
        candidate = path.name.split("-", 1)[0]
        header, body = parse_header(path.read_text(encoding="utf-8"))
        chunks.extend(
            chunk_markdown_document(
                body,
                candidate=candidate,
                source_type="platform",
                source_url=header.get("source") or header.get("source url", ""),
                fetched_date=header.get("fetched", ""),
                doc_title=header.get("title", path.stem),
                id_prefix=f"{candidate}-platform-{path.stem}",
            )
        )
    return chunks


def ingest_debates(corpus_dir: Path, fetched_date: str = "2026-10-23") -> list[Chunk]:
    chunks: list[Chunk] = []
    debate_dir = corpus_dir / "debates"
    if not debate_dir.exists():
        return chunks
    files = sorted(debate_dir.glob("*transcript*.txt")) + sorted(debate_dir.glob("*.vtt"))
    # The same debate may exist as both a .vtt and a cleaned .txt derived
    # from it. Ingest one representation per debate — the .vtt, which
    # carries cue timestamps and exact speaker-change boundaries.
    by_debate: dict[str, Path] = {}
    for path in files:
        key = re.sub(r"(-transcript|\.en(-orig)?)$", "", path.stem)
        if key not in by_debate or path.suffix == ".vtt":
            by_debate[key] = path
    for path in sorted(by_debate.values()):
        text = path.read_text(encoding="utf-8")
        turns = parse_vtt(text) if path.suffix == ".vtt" else parse_transcript_text(text)
        for turn in turns:
            turn["candidate"] = candidate_for_speaker(turn.get("speaker", "UNKNOWN"))
        chunks.extend(
            chunk_transcript(
                turns,
                source_url=f"local:{path.name}",
                fetched_date=fetched_date,
                doc_title=path.stem,
                id_prefix=f"debate-{path.stem}",
            )
        )
    return chunks


def ingest_records(corpus_dir: Path, fetched_date: str = "2026-10-23") -> list[Chunk]:
    records_dir = corpus_dir / "records"
    if not records_dir.exists():
        return []
    whole = records_dir / "member-voting-record-2022-2026.csv"
    if whole.exists():
        files = [whole]
    else:
        files = sorted(records_dir.glob("member-voting-record-2022-2026-part*.csv"))
    rows: list[dict] = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        return []
    return chunk_voting_records(
        rows,
        source_url="local:member-voting-record-2022-2026.csv",
        fetched_date=fetched_date,
        doc_title="Toronto Council Voting Record 2022-2026 (Toronto Open Data)",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="comparer.ingest")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--out", default="data/chunks.jsonl")
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    chunks = ingest_platforms(corpus_dir) + ingest_debates(corpus_dir) + ingest_records(corpus_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    stats = Counter((c.candidate, c.source_type) for c in chunks)
    print(f"wrote {len(chunks)} chunks to {out}")
    for (candidate, source_type), count in sorted(stats.items()):
        print(f"  {candidate:10s} {source_type:10s} {count}")


if __name__ == "__main__":
    main()
