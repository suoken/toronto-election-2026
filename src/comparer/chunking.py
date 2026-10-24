"""Chunking — the module the whole product's honesty rests on.

Owner's locked design decisions implemented here:

* **Platforms / documents:** structure-first splitting. A document is split
  on its heading hierarchy first; every chunk carries its ``heading_path``.
  A recursive size cap (~1200 chars) then splits *inside* a section on
  paragraph/sentence boundaries. A chunk never spans two sections.
* **Debate transcripts:** one speaker turn is the base chunk unit. A chunk
  NEVER spans two speakers, and speaker attribution is a hard invariant —
  when an over-long turn is split, every piece inherits the speaker tag.
  Topic-splitting inside a turn is *not* attempted (it is a measured
  optimisation for a later iteration, not a day-one feature).

The recursive splitter uses zero overlap on purpose: overlapping chunks
duplicate claims across chunk boundaries, which makes citations ambiguous.
Keeping pieces disjoint keeps every sentence attributable to exactly one
chunk.
"""

from __future__ import annotations

import html
import re

from .schema import Chunk

MAX_CHUNK_CHARS = 1200

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def recursive_split(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into pieces of at most ``max_chars`` characters.

    Tries paragraph breaks first, then sentence boundaries, then a hard
    character cut as a last resort. Pieces are packed greedily and are
    disjoint (no overlap).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Break into the finest units we are willing to cut on.
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            for sent in split_sentences(para):
                if len(sent) <= max_chars:
                    units.append(sent)
                else:  # pathological: one gigantic "sentence"
                    for i in range(0, len(sent), max_chars):
                        units.append(sent[i : i + max_chars])

    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = unit
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown_document(
    text: str,
    *,
    candidate: str,
    source_type: str,
    source_url: str,
    fetched_date: str,
    doc_title: str,
    id_prefix: str,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Chunk a markdown/plain document on its heading structure.

    Sections are delimited by ATX headings (``#``..``######``). Content
    before the first heading forms a section with ``heading_path`` equal
    to just the document title. Each section is then size-capped with
    :func:`recursive_split`.
    """
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def current_path() -> list[str]:
        return [title for _, title in heading_stack]

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_path(), body))
        buffer.clear()

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            buffer.append(line)
    flush()

    chunks: list[Chunk] = []
    for path, body in sections:
        for piece in recursive_split(body, max_chars=max_chars):
            chunks.append(
                Chunk(
                    id=f"{id_prefix}-{len(chunks):04d}",
                    text=piece,
                    candidate=candidate,
                    source_type=source_type,
                    source_url=source_url,
                    fetched_date=fetched_date,
                    doc_title=doc_title,
                    chunk_index=len(chunks),
                    heading_path=list(path),
                )
            )
    return chunks


# --------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------

# "CHOW:" / "Olivia Chow:" / "MODERATOR:" at the start of a line.
_SPEAKER_LINE_RE = re.compile(r"^\s*(?:\[[0-9:]+\]\s*)?([A-Z][A-Za-z .'-]{0,40}):\s+(.*)$")
_TIMESTAMP_PREFIX_RE = re.compile(r"^\[([0-9:]+)\]\s*")


class Turn(dict):
    """A single speaker turn: {speaker, timestamp, text, candidate}."""


def _is_speaker_label(candidate_label: str) -> bool:
    label = candidate_label.strip()
    if not label or len(label) > 40:
        return False
    words = label.split()
    if len(words) > 4:
        return False
    # ALL-CAPS label, or a capitalised name-like label.
    return label.upper() == label or all(w[0].isupper() for w in words if w)


def parse_transcript_text(text: str) -> list[Turn]:
    """Parse a plain-text transcript into speaker turns.

    Assumption (documented): turns are introduced by ``Name:`` at the start
    of a line, optionally preceded by a ``[HH:MM:SS]`` timestamp. Lines
    without a label continue the current turn. If no speaker labels exist
    anywhere in the document, each blank-line-separated block becomes its
    own turn attributed to ``"UNKNOWN"`` — we never merge blocks, because
    merging could silently span two speakers.
    """
    turns: list[Turn] = []
    current: Turn | None = None

    # Caption-derived text (e.g. cleaned YouTube transcripts) marks speaker
    # changes with inline ">>" but names nobody. Split on those markers so
    # a chunk can never span two speakers, even in this flattened form.
    raw_lines = text.splitlines()
    has_labels = any(
        (m := _SPEAKER_LINE_RE.match(ln.strip())) and _is_speaker_label(m.group(1))
        for ln in raw_lines
    )
    if ">>" in text and not has_labels:
        return [
            Turn(speaker="UNKNOWN", timestamp=None, text=segment, candidate="unknown")
            for segment in (
                re.sub(r"\s+", " ", s).strip() for s in text.split(">>")
            )
            if segment
        ]

    def flush() -> None:
        nonlocal current
        if current is not None and current["text"].strip():
            current["text"] = re.sub(r"\s+", " ", current["text"]).strip()
            turns.append(current)
        current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _SPEAKER_LINE_RE.match(line)
        if match and _is_speaker_label(match.group(1)):
            flush()
            ts_match = _TIMESTAMP_PREFIX_RE.match(line)
            current = Turn(
                speaker=match.group(1).strip(),
                timestamp=ts_match.group(1) if ts_match else None,
                text=match.group(2),
                candidate="unknown",
            )
        elif current is not None:
            current["text"] += " " + line
        elif line:
            current = Turn(speaker="UNKNOWN", timestamp=None, text=line, candidate="unknown")
    flush()

    if not turns:
        for block in re.split(r"\n\s*\n", text):
            block = re.sub(r"\s+", " ", block).strip()
            if block:
                turns.append(Turn(speaker="UNKNOWN", timestamp=None, text=block, candidate="unknown"))
    return turns


def parse_vtt(text: str) -> list[Turn]:
    """Parse a YouTube WebVTT caption file into speaker turns.

    YouTube auto-captions mark a change of speaker with a leading ``>>``
    but never identify *who* is speaking, so every turn's speaker is
    ``"UNKNOWN"``; turns are still kept separate because a ``>>`` boundary
    is a real speaker boundary. The rolling-caption duplication (each cue
    repeats the previous line while the next line is being spoken) is
    collapsed by dropping lines identical to, or strict prefixes of, the
    previous emitted line.
    """
    lines_out: list[tuple[str | None, str]] = []  # (cue start timestamp, line)
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        cue_lines = [ln for ln in block.splitlines() if ln.strip()]
        if not cue_lines:
            continue
        timing_idx = next((i for i, ln in enumerate(cue_lines) if "-->" in ln), None)
        if timing_idx is None:
            continue
        start = cue_lines[timing_idx].split("-->")[0].strip()
        timestamp = start.rsplit(".", 1)[0]  # HH:MM:SS
        for raw in cue_lines[timing_idx + 1 :]:
            clean = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
            if clean:
                lines_out.append((timestamp, clean))

    # Collapse rolling duplication.
    emitted: list[tuple[str | None, str]] = []
    for ts, line in lines_out:
        if emitted:
            prev = emitted[-1][1]
            if line == prev or prev.startswith(line):
                continue
            if line.startswith(prev) and len(emitted) >= 1:
                # New line extends the previous partial line: replace it.
                emitted[-1] = (emitted[-1][0], line)
                continue
        emitted.append((ts, line))

    turns: list[Turn] = []
    current: Turn | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current["text"].strip():
            current["text"] = re.sub(r"\s+", " ", current["text"]).strip()
            turns.append(current)
        current = None

    for ts, line in emitted:
        if line.startswith(">>"):
            flush()
            current = Turn(speaker="UNKNOWN", timestamp=ts, text=line.lstrip(">").strip(), candidate="unknown")
        elif current is None:
            current = Turn(speaker="UNKNOWN", timestamp=ts, text=line, candidate="unknown")
        else:
            current["text"] += " " + line
    flush()
    return turns


def chunk_transcript(
    turns: list[Turn],
    *,
    source_url: str,
    fetched_date: str,
    doc_title: str,
    id_prefix: str,
    source_type: str = "debate",
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Chunk parsed turns. One turn = one chunk, unless the turn exceeds
    the size cap — then it is split *within the turn only*, and every piece
    inherits the turn's speaker and candidate. A chunk never spans turns,
    so it can never span speakers."""
    chunks: list[Chunk] = []
    for turn in turns:
        for piece in recursive_split(turn["text"], max_chars=max_chars):
            chunks.append(
                Chunk(
                    id=f"{id_prefix}-{len(chunks):04d}",
                    text=piece,
                    candidate=turn.get("candidate", "unknown"),
                    source_type=source_type,
                    source_url=source_url,
                    fetched_date=fetched_date,
                    doc_title=doc_title,
                    chunk_index=len(chunks),
                    speaker=turn.get("speaker"),
                    timestamp=turn.get("timestamp"),
                )
            )
    return chunks
