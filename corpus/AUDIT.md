# Corpus Audit — Toronto Candidate Policy Comparer

**Date:** October 23–24, 2026 (overnight build) · **Rules set by Ken:** public records only; paywalled sources skipped outright, never fetched.
**Companion:** `corpus/MANIFEST.json` (per-file checksums, sizes, word counts).

## Summary

The corpus is real, public, and usable tonight — with two honest gaps: five of six debate videos are still blocked behind YouTube bot mitigation (retry scheduled), and debate chunks cannot yet be attributed to speakers because YouTube captions never name them. (The voting-record tier was chunked on 2026-10-25 — see below.)

## What exists

### Platforms — complete (93 files, ~67,000 words)
| Candidate | Files | Words | Shape |
|---|---|---|---|
| Olivia Chow (oliviachow.ca) | 40 | ~22,700 | One consolidated plan page (`/plan`, 15 topics in full) + 31 announcement posts + 8 position landing pages |
| Brad Bradford (bradford26.ca) | 53 | ~44,200 | **No platform document exists.** Homepage names 5 priorities; all commitments live in 52 press releases (May–Oct 2026), all saved |

- Neither campaign publishes a platform PDF. Web text is the entire platform record.
- **Structural caveat:** Bradford's positions are time-ordered announcements — later releases supersede earlier ones. His chunks do not yet carry publication dates in metadata, so recency down-weighting is documented-but-not-implemented.
- Chow utility pages (donate/volunteer forms) were fetched and deliberately excluded — no policy text.

### Debates — 1 of 7 transcripts in hand
- **In corpus:** TVO/Rotman debate, Oct 15, 2026 (Paikin moderating) — 10,820-word transcript from YouTube **auto-captions** (no manual track exists). Speaker changes are marked (`>>`) but speakers are **never named**, so all 231 debate chunks are candidate=`unknown`: searchable, but per-candidate answers come from platforms only. Auto-caption quality is good in spot checks; proper nouns/numbers need spot-verification before citation.
- **Blocked (captions confirmed to exist, download refused):** CP24 "Toronto Decides" final debate (Oct 22), CityNews debates (Sept 22, Oct 6, Oct 7), CBC debate (Oct 9 upload), Toronto Star-posted video (Oct 7 upload). YouTube flagged this machine's IP mid-task ("Sign in to confirm you're not a bot"). No published text transcripts exist for these — only recaps.
- **Recovery:** one slow retry pass scheduled 2026-10-24 03:30 ET (one attempt per video, spaced; stops if the block persists). If still blocked, the remaining route is a signed-in browser session with Ken.

### Voting records — acquired and chunked (2026-10-25)
- Toronto Open Data, "Members of Toronto City Council – Voting Record": `corpus/records/member-voting-record-2022-2026.csv` — 46,815 vote rows, 2022-11-23 → 2026-08-05, 30 members, fields include agenda item # (motion id), date/time, member, vote, result. Chow: 1,420 rows; Bradford: 1,836 rows.
- Chunked 2026-10-25: one chunk per candidate per vote-event (3,256 record chunks), each stating only that candidate's own vote; absences are phrased as absences, never as No votes. The golden eval set has no record questions yet, so record retrieval is verified live, not by the eval suite.

### News & social — inventoried, not bulk-collected (by design)
- Free and covering the race: CBC Toronto, CP24, CTV Toronto, Global News, CityNews, TVO.
- **Skipped per Ken's rule:** Toronto Star, Globe and Mail, National Post, Toronto Sun (metered).
- Candidate social accounts mapped in `corpus/source-inventory-news-social.md` (Chow: @oliviachow campaign vs. mayor's-office accounts; Bradford: @BradMBradford / @bradfordgrams / TikTok @bradbradfordto).

## Build state (overnight, Oct 23–24)

- **3,894 chunks** in `data/chunks.jsonl`: Chow platform 130, Bradford platform 277, TVO debate (unattributed) 231, council voting records 3,256 (Chow 1,420, Bradford 1,836).
- **Embeddings: hashing fallback.** sentence-transformers could not be installed (the ~554 MB torch wheel failed to download 3× on network errors). A deterministic 4,096-dim hashing embedder is live; one successful `pip install sentence-transformers` + re-embed switches to MiniLM.
- **Generation: extractive mode.** No LLM API key is configured, so answers are composed from verbatim cited sentences (no synthesis, no difference summary). Adding `ANTHROPIC_API_KEY` to `.env` switches to the LLM path with the same critic pass; that path is written but untested.
- **Tests:** 7 passed (speaker-span invariant, heading paths, retrieval balance under 20:2 volume skew, critic vs. unsupported/fake/misattributed citations, API).

### Eval results (golden set: 31 questions — 13 facts, 9 contested, 3 "not addressed" probes, 6 refusals)
| Metric | Result | Caveat |
|---|---|---|
| Faithfulness | 100% (124/124 claims) | Partly structural: extractive claims are verbatim sentences carrying their own citations. Expect this to drop — and start meaning more — when the LLM path paraphrases. |
| Balance coverage | 100% (9/9) | On the golden set. |
| Refusal correctness | 100% (6/6), 0 false refusals | Includes "Who should I vote for?" → refused, per the product boundary. |
| Per-candidate status accuracy | 100% (47/47) | Required a real fix: an IDF-weighted coverage gate (≥50% of question vocabulary) to separate "Chow on the land transfer tax" (absent) from "Chow on property taxes" (present) — raw similarity could not. |
| Retrieval hit-rate (proxy) | 100% (22/22) | Golden-set calibrated; a held-out set is needed before treating any of these as generalization. |

## Gaps & next steps (in order)
1. Debate transcripts ×6 — retry 2026-10-24 03:30 ET; fallback is a signed-in browser session.
2. Speaker attribution for debate chunks — captions don't name speakers; needs either diarization or manual labeling of turn blocks before debates can feed per-candidate answers.
3. ~~Voting-record chunker~~ — done 2026-10-25 (3,256 record chunks). Next for this tier: record questions in the golden set.
4. Bradford publication dates into chunk metadata → recency handling for superseded positions.
5. sentence-transformers install on a healthy connection → re-embed, re-run evals.
6. LLM API key (Ken's decision) → enables synthesized answers + difference summaries; re-run evals and expect faithfulness to become a real measurement.
7. Held-out golden set #2, authored after #1 is frozen, to validate the 100% scores honestly.
