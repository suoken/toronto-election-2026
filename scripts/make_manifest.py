#!/usr/bin/env python3
"""Build corpus/MANIFEST.json — an index of every corpus source file.

For each file under the corpus dir: relative path, size in bytes, sha256,
and (for .md/.txt files) a word count. Plus a summary block with the file
count, total words, and per-subfolder counts.

This is inventory plumbing for the M1 corpus audit. It does not parse,
chunk, or interpret the sources in any way.

Usage: python scripts/make_manifest.py [corpus_dir]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    corpus_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else repo_root / "corpus"
    if not corpus_dir.is_dir():
        raise SystemExit(f"corpus dir not found: {corpus_dir}")

    files = []
    per_subfolder: dict[str, dict[str, int]] = {}
    total_words = 0

    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        rel = path.relative_to(corpus_dir)
        entry: dict = {
            "path": rel.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }
        if path.suffix.lower() in TEXT_SUFFIXES:
            words = len(path.read_text(encoding="utf-8", errors="replace").split())
            entry["words"] = words
            total_words += words
        files.append(entry)

        subfolder = rel.parts[0] if len(rel.parts) > 1 else "."
        bucket = per_subfolder.setdefault(subfolder, {"files": 0, "words": 0})
        bucket["files"] += 1
        bucket["words"] += entry.get("words", 0)

    manifest = {
        "corpus_dir": corpus_dir.name,
        "summary": {
            "file_count": len(files),
            "total_words": total_words,
            "per_subfolder": per_subfolder,
        },
        "files": files,
    }

    out = corpus_dir / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} — {len(files)} files, {total_words} words")


if __name__ == "__main__":
    main()
