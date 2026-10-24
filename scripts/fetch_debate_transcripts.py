#!/usr/bin/env python3
"""
Fetch auto-caption transcripts for the six blocked Toronto debate videos.

Uses the yt_dlp Python library if it's installed, otherwise falls back to
the `yt-dlp` command-line tool. Downloads captions only (no video), then
converts each .vtt file into clean plain text with the rolling-caption
duplicates removed.

Usage:
    python3 fetch_debate_transcripts.py
    python3 fetch_debate_transcripts.py --out ~/debate-transcripts
    python3 fetch_debate_transcripts.py --cookies-from-browser chrome

Output (default folder: ./debate-transcripts):
    <slug>.en.vtt            raw captions, kept as downloaded
    <slug>-transcript.txt    cleaned transcript, this is the file to share back

Nothing is uploaded anywhere. Everything runs and stays on your machine.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

VIDEOS = [
    ("cp24-toronto-decides-2026-10-22", "https://www.youtube.com/watch?v=GkgepW94n08"),
    ("citynews-debate-2026-10-06",      "https://www.youtube.com/watch?v=M-ai6um2Roc"),
    ("citynews-debate-2026-10-07",      "https://www.youtube.com/watch?v=W7PbrxypXQY"),
    ("citynews-debate-2026-09-22",      "https://www.youtube.com/watch?v=tw8_zSXR32I"),
    ("cbc-debate-2026-10-09",           "https://www.youtube.com/watch?v=U5PDQ3u7MUk"),
    ("star-video-2026-10-07",           "https://www.youtube.com/watch?v=IdCgvsgeFQo"),
]


def download_captions(slug, url, out_dir, cookies_browser=None):
    """Download auto-captions for one video. Returns the .vtt path or None."""
    outtmpl = os.path.join(out_dir, f"{slug}.%(ext)s")

    try:
        import yt_dlp

        opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en.*"],
            "subtitlesformat": "vtt",
            "outtmpl": outtmpl,
            "quiet": False,
        }
        if cookies_browser:
            opts["cookiesfrombrowser"] = (cookies_browser,)
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except ImportError:
        cmd = [
            "yt-dlp", "--skip-download", "--write-auto-subs",
            "--sub-langs", "en.*", "--sub-format", "vtt",
            "-o", outtmpl,
        ]
        if cookies_browser:
            cmd += ["--cookies-from-browser", cookies_browser]
        cmd.append(url)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return None

    vtts = sorted(glob.glob(os.path.join(out_dir, f"{slug}*.vtt")))
    return vtts[0] if vtts else None


def vtt_to_clean_text(vtt_path):
    """Strip VTT structure/tags and collapse rolling-caption duplicates."""
    lines = []
    with open(vtt_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
                continue
            if "-->" in line:  # timing line
                continue
            line = re.sub(r"<[^>]+>", "", line)  # <c>, timestamps, etc.
            line = line.strip()
            if not line:
                continue
            # Rolling captions repeat/grow lines: if the previous kept line is
            # contained in this one, this one supersedes it; if this one is
            # contained in the previous, it's a repeat — skip it.
            if lines and lines[-1] in line:
                lines[-1] = line
            elif lines and line in lines[-1]:
                continue
            elif lines and line == lines[-1]:
                continue
            else:
                lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="debate-transcripts",
                        help="output folder (default: ./debate-transcripts)")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                        help="only if YouTube still blocks you, e.g. chrome — "
                             "uses your signed-in session for the download")
    parser.add_argument("--sleep", type=int, default=10,
                        help="seconds between videos (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = []

    for i, (slug, url) in enumerate(VIDEOS):
        if i:
            time.sleep(args.sleep)
        print(f"\n=== {slug} ===")
        try:
            vtt = download_captions(slug, url, args.out, args.cookies_from_browser)
        except Exception as e:  # yt-dlp raises DownloadError on bot check / 429
            print(f"FAILED: {e}")
            results.append((slug, "FAILED — see error above"))
            continue
        if not vtt:
            print("FAILED: no caption file was produced")
            results.append((slug, "FAILED — no captions"))
            continue

        text = vtt_to_clean_text(vtt)
        txt_path = os.path.join(args.out, f"{slug}-transcript.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        words = len(text.split())
        print(f"OK: {txt_path} ({words:,} words)")
        results.append((slug, f"OK — {words:,} words"))

    print("\n--- Summary ---")
    for slug, status in results:
        print(f"{slug}: {status}")


if __name__ == "__main__":
    sys.exit(main())
