# Toronto Candidate Policy Comparer

Ask a question about the 2026 Toronto mayoral candidates and get a
side-by-side answer built **only from cited public sources** — campaign
platforms, debate transcripts, and the council voting record.

The product boundary is permanent: **it compares, it never recommends.**
Ask it "who should I vote for?" and it refuses, by design.

Built as a portfolio project in applied AI engineering: the interesting
problems here are not the LLM calls, they are retrieval balance,
citation faithfulness, misattribution prevention, and measuring all
three with an eval harness.

## Status

- Core pipeline built and evaluated (October 2026, ahead of the
  October 26 election): ingestion, chunking, hybrid retrieval,
  generation, critic pass, and a 31-question golden-set eval.
- Runs in **extractive mode** by default: answers are composed from
  verbatim cited sentences, with local embeddings. Setting an LLM API
  key in `.env` switches generation to a provider model (the interface
  is provider-swappable); that path is written but not yet eval'd.
- Corpus: 638 chunks — Chow platform 130, Bradford platform 277,
  TVO debate 231 (see "Known limits" for why debate chunks are
  unattributed).

## Eval results

Latest run (`evals/latest-report.json`, golden set: 13 fact questions,
9 contested questions, 3 "not addressed" probes, 6 refusal cases):

| Metric | Result |
|---|---|
| Faithfulness (claims supported by their citation) | 100% (124/124) |
| Balance coverage (both candidates answered on contested items) | 100% (9/9) |
| Refusal correctness (incl. "Who should I vote for?") | 100% (6/6) |
| Per-candidate status accuracy | 100% (47/47) |
| Retrieval hit rate | 100% (22/22) |

Read these honestly: in extractive mode, faithfulness is partly
structural — claims are verbatim sentences carrying their own
citations. The metric will do real work (and likely drop) on the LLM
path. The golden set was also calibrated against itself; a held-out
set is on the roadmap before these numbers mean generalization.

## How it works

1. **Ingest & chunk** (`src/comparer/chunking.py`) — structure-first:
   platform documents split on headings, debate transcripts split on
   speaker turns. **One speaker turn is the chunk unit for transcripts,
   and speaker attribution is a hard invariant** — a chunk never spans
   two speakers, because a citation that attributes one candidate's
   words to the other is the worst failure this product can make.
2. **Retrieve** (`retrieval.py`) — hybrid: cosine similarity + BM25,
   fused with reciprocal rank fusion, retrieved **per candidate** and
   then merged, so the candidate with more published material cannot
   drown out the other.
3. **Decide "not addressed"** — an IDF-weighted coverage gate: a
   candidate counts as having addressed an issue only if retrieved
   chunks cover enough of the question's distinctive vocabulary. (Raw
   similarity alone could not separate "property taxes" — covered —
   from "land transfer tax" — absent — for the same candidate.)
4. **Generate** (`generation.py`) — provider interface; extractive
   verbatim answers by default, LLM generation when a key is present.
5. **Critic pass** (`critic.py`) — every claim is checked: does its
   citation resolve, is it attributed to the right candidate, is it
   supported by the cited chunk? Failing claims are removed, not
   shipped.
6. **Answer** (`answer.py`) — classify the question, gate per
   candidate, retrieve, generate, critic-check, and return a
   structured result with a full trace. Served at `POST /ask`.

## Corpus

Public records only — if a source was paywalled, it was skipped.

- `corpus/platforms/` — 93 files (~67,000 words). Chow has one
  consolidated plan page plus announcements; Bradford has no platform
  document — his commitments are scattered across 52 press releases,
  so recency ordering matters.
- `corpus/debates/` — the TVO/Rotman October 15 debate transcript
  (auto-captions). Six further debate videos are blocked by YouTube
  bot mitigation from this network; `scripts/fetch_debate_transcripts.py`
  fetches and cleans their captions from any ordinary connection.
- `corpus/records/` — Toronto Open Data council voting record,
  2022–2026 term (46,815 rows). The CSV is stored split into four
  parts; see `corpus/records/README.md` to recombine it.
- `corpus/AUDIT.md` — full corpus audit: scope, gaps, and method.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m comparer.ingest        # build chunks from the corpus
python -m comparer.store load    # load chunks into the store
uvicorn comparer.main:app        # POST /ask, GET /health

pytest                           # 7 tests
python evals/run_evals.py        # golden-set eval → evals/latest-report.json
```

Configuration is via `.env` (see `.env.example`): database URL,
optional LLM API key, embedding model, monthly cost cap. The default
embedder is a deterministic hashing embedder; installing the `local`
extra (sentence-transformers) and re-embedding switches to MiniLM.

## Known limits

- Debate chunks are `candidate = unknown`: the TVO auto-captions mark
  speaker changes but never name speakers, so per-candidate answers
  currently come from platforms. Speaker attribution for debates is
  the top roadmap item, along with the remaining debate transcripts,
  a voting-record chunker (the record tier is schema-ready but not yet
  ingested), Bradford publication dates in chunk metadata, and a
  held-out golden set.
- Post-election direction: the same pipeline over council records —
  "how did my councillor vote on X?" — so the project outlives the
  campaign.

## Layout

```
src/comparer/   pipeline: schema, chunking, ingest, store, embeddings,
                retrieval, prompts, generation, critic, answer, API
corpus/         platforms, debates, records, audit
evals/          golden set + eval harness + latest report
tests/          chunking, retrieval, critic, health
scripts/        corpus manifest builder; YouTube transcript fetcher
sections/, voter-guide.md   the hand-built voter guide the corpus grew from
```
