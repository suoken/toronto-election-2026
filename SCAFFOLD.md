# Scaffold — Toronto Candidate Policy Comparer

This document describes the project scaffolding: the plumbing that exists
so the real modules can be built on top of it. It was created on
2026-10-23 as setup work only — none of the product's core logic is here.

## What was scaffolded

| Path | What it is |
|---|---|
| `pyproject.toml` | Project `toronto-comparer` (Python ≥3.11, src layout). Runtime deps: fastapi, uvicorn, pydantic, pydantic-settings, python-dotenv. Dev deps: pytest, httpx. |
| `docker-compose.yml` | A single service: Postgres 16 with pgvector (`pgvector/pgvector:pg16`) on port 5432, env-driven credentials with local defaults, named volume `comparer_pgdata`. Nothing else. |
| `src/comparer/config.py` | `Settings` via pydantic-settings, read from `.env`: `database_url`, `llm_api_key` (optional), `embedding_model`, `monthly_cost_cap_usd` (default 50), `corpus_dir`. |
| `src/comparer/main.py` | FastAPI app shell. `GET /health` → `{"status": "ok"}`. `POST /ask` accepts `{"question": str}` and returns HTTP 501 `{"detail": "answering pipeline not built yet"}`. No answering logic. |
| `tests/test_health.py` | Pytest smoke tests (FastAPI TestClient): `/health` is 200, `/ask` is 501. |
| `.github/workflows/ci.yml` | On push/PR: Python 3.12, `pip install -e ".[dev]"`, run `pytest`. |
| `scripts/make_manifest.py` | Walks the corpus dir and writes `corpus/MANIFEST.json` (per-file path, size, sha256, word count for .md/.txt; summary with totals and per-subfolder counts). Inventory only — it does not parse or interpret sources. |
| `.env.example` | All config keys with placeholder values. Copy to `.env` (gitignored). |
| `.gitignore` | Standard Python, `.env`, `.venv`. |

## How to run

```bash
cd toronto-election-2026

# 1. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then fill in real values

# 2. Database (Postgres + pgvector)
docker compose up -d

# 3. API shell
uvicorn comparer.main:app --app-dir src --reload
# -> GET  http://127.0.0.1:8000/health
# -> POST http://127.0.0.1:8000/ask   (501 until the pipeline exists)

# 4. Tests
pytest

# 5. Corpus manifest
python scripts/make_manifest.py     # rewrites corpus/MANIFEST.json
```

## Intentionally not built

The following are **owner-built modules** — the load-bearing parts of the
product, deliberately left for the project owner to design and write.
The scaffolding must not grow into them:

- **Chunk schema + transcript/document chunker** — the metadata shape
  (candidate, source type, date, URL, structural location) and the
  structure-first splitting logic.
- **Retrieval layer** — hybrid vector + keyword search, per-candidate
  balanced retrieval, reranking, and its retrieval eval.
- **Generation prompts + critic pass** — the constrained answering prompt
  and the claim-verification step.
- **Eval harness** — faithfulness / balance / refusal / regression suites
  and their CI gates.
- **Golden set** — the ~50 questions with expected citations and behaviors.
- **Chunks database table** — the storage schema for chunks; it mirrors
  the owner's chunk schema, so it is designed after that schema exists.

Until those land, `POST /ask` stays a 501 on purpose: a shell that
pretends to answer would be worse than one that honestly doesn't.

---

## Core modules (added 2026-10-24 — supersedes "Intentionally not built")

On 2026-10-23 the owner reversed the tutor-mode reservation and directed
that the full core be built ("just go for the best option"). The modules
listed above as intentionally not built now exist:

| Path | What it is |
|---|---|
| `src/comparer/schema.py` | `Chunk` dataclass: id, text, candidate, source_type (platform/debate/record/news), source_url, fetched_date, doc_title, heading_path (or speaker+timestamp), chunk_index, char_count; evidence-tier mapping (claim / voting record / news reporting). |
| `src/comparer/chunking.py` | Structure-first markdown chunking (heading paths, ~1200-char recursive cap inside sections, zero overlap). Transcript handling: one speaker turn = one chunk, chunks never span speakers, over-long turns split within the turn and inherit the speaker. Includes a `Name:` turn parser and a YouTube WebVTT parser (`>>` = speaker change, speaker unidentified). |
| `src/comparer/ingest.py` | CLI: `python -m comparer.ingest` — reads `corpus/platforms/*.md` (headers parsed; candidate from filename) and `corpus/debates/` (`.vtt` preferred over a derived `-transcript.txt` for the same debate), writes `data/chunks.jsonl`. |
| `src/comparer/store.py` | SQLAlchemy `chunks` table mirroring the schema; SQLite by default, `DATABASE_URL` override for the Postgres compose service. Embeddings stored as a JSON column (pgvector migration noted as TODO in the module). CLI: `python -m comparer.store load --embed`. |
| `src/comparer/embeddings.py` | Provider interface. Default: local sentence-transformers `all-MiniLM-L6-v2`. Fallback: deterministic hashing embedder (4096-dim, stopword-filtered). **The local model could not be installed on 2026-10-24** — the torch wheel download failed three times on network errors — so everything below ran on the hashing embedder. `EMBEDDER=hashing` forces the fallback. |
| `src/comparer/retrieval.py` | Hybrid search: cosine over embeddings + BM25 (rank_bm25, with a built-in fallback implementation), fused with RRF. Retrieval is per candidate (top-k each), then merged — balance is architectural, not prompted. |
| `src/comparer/prompts.py` | The constrained generation prompt: cite every claim from provided chunks only, "not addressed" is a first-class per-candidate outcome, evidence tiers labelled, compare-never-recommend. |
| `src/comparer/generation.py` | Provider interface. With `ANTHROPIC_API_KEY` (preferred) or `OPENAI_API_KEY` in the env: LLM generation into a fixed JSON shape. **No key is currently configured**, so the live path is the extractive generator: verbatim, question-relevant sentences from retrieved chunks, each with its citation; `difference_summary` stays null (no synthesis without an LLM). |
| `src/comparer/critic.py` | Verification pass: every claim's citations must resolve to real chunks, must not cite the other candidate's chunks (misattribution), and must clear an embedding-support threshold against the cited chunk. Flagged claims are removed, not warned about. |
| `src/comparer/answer.py` | The pipeline behind `POST /ask`: rule-based question classification (fact / comparison / difference / refusal), per-candidate relevance + coverage gating (a candidate is only "answered" if their chunks actually cover the question's distinctive vocabulary, IDF-weighted), retrieval, generation, critic pass, and a structured response with a full trace. `POST /ask` now returns real answers (200), or 503 until ingest has run. |
| `evals/golden.jsonl` | 31 golden questions grounded in the actual corpus: 13 single-candidate facts, 9 contested comparisons, 3 verified "not addressed" probes, 6 refusal questions. |
| `evals/run_evals.py` | Harness: runs the pipeline over the golden set; writes `evals/latest-report.json`. |

### Pipeline runbook

```bash
.venv/bin/python -m comparer.ingest            # corpus -> data/chunks.jsonl
.venv/bin/python -m comparer.store load --embed  # -> SQLite + embeddings
.venv/bin/python -m evals.run_evals            # golden-set metrics
uvicorn comparer.main:app --app-dir src        # POST /ask is live
```

### Eval results (2026-10-24, hashing embedder + extractive generation)

| Metric | Result | Gate |
|---|---|---|
| Faithfulness (delivered claims supported + cited) | 100.0% (124/124) | ≥ 90% |
| Balance coverage (contested questions) | 100.0% (9/9) | 100% |
| Refusal correctness | 100.0% (6/6) | ≥ 95% |
| False refusals | 0.0% (0/25) | — |
| Per-candidate status accuracy | 100.0% (47/47) | — |
| Retrieval hit-rate (keyword proxy) | 100.0% (22/22) | ≥ 85% |
| Critic removals | 0 of 124 claims | — |

Read the faithfulness number with its context: on the extractive path,
claims are verbatim source sentences carrying their own citations, so
support is near-guaranteed by construction and the critic has little to
remove. These metrics will do real work — and will likely drop — when an
LLM starts paraphrasing. The coverage-gate thresholds were calibrated
while developing against this golden set; a held-out question set is
needed before treating 100% as a generalisation claim.

### What changes when an API key is added

Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) in `.env` and restart:
generation switches from extractive to the LLM path — synthesised
per-candidate answers and a real difference summary, with the same
citation constraints and the same critic pass (which will start earning
its keep). No code changes. Separately, once the network allows it,
`pip install sentence-transformers` enables the semantic embedder
(default when `EMBEDDER` is unset); re-run `store load --embed` and the
evals afterwards.

### Known limitations

- **Embeddings tonight are lexical, not semantic** (hashing fallback;
  see above). Retrieval still passed the golden set because BM25 + the
  coverage gate carry the distinctive-vocabulary load, but paraphrase
  questions will underperform until MiniLM is installed.
- **Debate chunks are unattributed.** YouTube auto-captions mark speaker
  *changes* (`>>`) but never identify speakers, so all 231 TVO debate
  chunks carry candidate `unknown`: indexed and searchable via the
  retriever, but the /ask pipeline answers per candidate from platform
  material only. A named transcript or diarization is required before
  debate evidence can be cited per candidate. The CP24 Oct 22 debate is
  not yet in the corpus at all.
- **Voting-record CSV is not ingested.** `corpus/records/` (46,815
  council votes) is acquired but has no chunker yet, so the "voting
  record" evidence tier is schema-ready but empty in practice.
- **Extractive sentence selection is lexical.** Occasionally a cited,
  supported, but off-angle sentence is chosen (e.g. a grocers by-law
  sentence surfacing for a property-tax question). Citations remain
  accurate; relevance polish is an LLM-path benefit.
- **Corpus imbalance is real:** Bradford has ~2.1× Chow's platform chunks
  (277 vs 130) because his commitments live in 52 separate releases.
  Per-candidate retrieval neutralises this for ranking, and Bradford's
  chunks carry `Published:` dates in their source files — but chunk
  metadata does not yet include the publication date, so superseded
  early-campaign positions are not yet down-weighted. Recency handling
  is future work.
- Refusal classification is rule-based (regex), tuned to the golden set;
  paraphrased recommendation-seeking may slip it.
