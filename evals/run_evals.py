"""Golden-set eval harness.

Runs the real pipeline over ``evals/golden.jsonl`` and reports the four
launch-gate metrics from the PRD, measured on the answers a user would
actually receive (i.e. after the critic pass):

* **Faithfulness** — share of delivered claims whose citations resolve to
  real chunks AND whose text is supported by a cited chunk (embedding
  similarity above the critic threshold). Independently recomputed here,
  not taken from the critic's own report.
* **Balance coverage** — on questions where both candidates are expected
  to have coverage, the share where both are actually represented.
* **Refusal correctness** — refusal questions refused; non-refusal
  questions not refused.
* **Retrieval hit-rate (proxy)** — for answerable questions, the share
  where at least one chunk retrieved for an expected candidate contains
  an expected keyword. A proxy: it shows retrieval surfaced the right
  material, not that the final wording is perfect.

Also reported: per-candidate status accuracy (expected answered /
not_addressed vs actual) and the critic's claim-removal rate.

    python -m evals.run_evals [--chunks data/chunks.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comparer.answer import Pipeline  # noqa: E402
from comparer.critic import SUPPORT_THRESHOLD  # noqa: E402
from comparer.embeddings import cosine  # noqa: E402


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evals.run_evals")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--golden", default="evals/golden.jsonl")
    parser.add_argument("--report", default="evals/latest-report.json")
    args = parser.parse_args(argv)

    pipeline = Pipeline.from_jsonl(args.chunks)
    embedder = pipeline.embedder
    golden = [
        json.loads(line)
        for line in Path(args.golden).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    totals = dict(
        claims=0, claims_faithful=0,
        balance_items=0, balance_ok=0,
        refusal_items=0, refusal_ok=0, nonrefusal_items=0, nonrefusal_ok=0,
        status_checks=0, status_ok=0,
        hit_items=0, hit_ok=0,
        critic_checked=0, critic_flagged=0,
    )
    per_item: list[dict] = []

    for item in golden:
        result = pipeline.answer(item["question"])
        entry: dict = {"id": item["id"], "type": item["type"], "problems": []}

        # --- refusals ---------------------------------------------------
        if item["expected_behavior"] == "refuse":
            totals["refusal_items"] += 1
            ok = result["refused"]
            totals["refusal_ok"] += ok
            if not ok:
                entry["problems"].append("expected refusal, got an answer")
        else:
            totals["nonrefusal_items"] += 1
            ok = not result["refused"]
            totals["nonrefusal_ok"] += ok
            if not ok:
                entry["problems"].append(f"unexpected refusal: {result['refusal_reason']}")

        # --- per-candidate status accuracy ------------------------------
        for cand in item["expected_answered"]:
            totals["status_checks"] += 1
            actual = result["per_candidate"][cand]["status"]
            totals["status_ok"] += actual == "answered"
            if actual != "answered":
                entry["problems"].append(f"{cand}: expected answered, got {actual}")
        for cand in item["expected_not_addressed"]:
            totals["status_checks"] += 1
            actual = result["per_candidate"][cand]["status"]
            totals["status_ok"] += actual == "not_addressed"
            if actual != "not_addressed":
                entry["problems"].append(f"{cand}: expected not_addressed, got {actual}")

        # --- balance coverage -------------------------------------------
        if set(item["expected_answered"]) == {"chow", "bradford"}:
            totals["balance_items"] += 1
            both = all(
                result["per_candidate"][c]["status"] == "answered"
                for c in ("chow", "bradford")
            )
            totals["balance_ok"] += both
            if not both:
                entry["problems"].append("balance: one candidate missing")

        # --- faithfulness of delivered claims (independent recompute) ---
        for cand, cand_answer in result["per_candidate"].items():
            for claim in cand_answer["claims"]:
                totals["claims"] += 1
                cited = [
                    pipeline.chunks_by_id[cid]
                    for cid in claim["citations"]
                    if cid in pipeline.chunks_by_id
                ]
                if not claim["citations"] or not cited:
                    entry["problems"].append(f"{cand}: claim with unresolvable citation")
                    continue
                claim_vec = embedder.embed_one(claim["text"])
                best = max(cosine(claim_vec, embedder.embed_one(c.text)) for c in cited)
                if best >= SUPPORT_THRESHOLD:
                    totals["claims_faithful"] += 1
                else:
                    entry["problems"].append(f"{cand}: claim support {best:.2f} below threshold")

        # --- retrieval hit-rate proxy ------------------------------------
        keywords = [k.lower() for k in item.get("keywords", [])]
        if keywords and item["expected_answered"]:
            totals["hit_items"] += 1
            hit = False
            for cand in item["expected_answered"]:
                for cid in result["trace"]["chunks_retrieved"].get(cand, []):
                    if any(k in pipeline.chunks_by_id[cid].text.lower() for k in keywords):
                        hit = True
            totals["hit_ok"] += hit
            if not hit:
                entry["problems"].append("retrieval: no expected keyword in retrieved chunks")

        critic = result["trace"].get("critic") or {}
        totals["critic_checked"] += critic.get("claims_checked", 0)
        totals["critic_flagged"] += critic.get("claims_flagged", 0)
        per_item.append(entry)

    metrics = {
        "embedder": getattr(embedder, "name", "unknown"),
        "generator": getattr(pipeline.generator, "name", "unknown"),
        "questions": len(golden),
        "faithfulness_pct": pct(totals["claims_faithful"], totals["claims"]),
        "claims_delivered": totals["claims"],
        "claims_faithful": totals["claims_faithful"],
        "balance_coverage_pct": pct(totals["balance_ok"], totals["balance_items"]),
        "balance_items": totals["balance_items"],
        "refusal_correct_pct": pct(totals["refusal_ok"], totals["refusal_items"]),
        "false_refusal_rate_pct": pct(
            totals["nonrefusal_items"] - totals["nonrefusal_ok"], totals["nonrefusal_items"]
        ),
        "status_accuracy_pct": pct(totals["status_ok"], totals["status_checks"]),
        "retrieval_hit_rate_pct": pct(totals["hit_ok"], totals["hit_items"]),
        "critic_claims_checked": totals["critic_checked"],
        "critic_claims_flagged": totals["critic_flagged"],
    }

    print(f"Golden set: {metrics['questions']} questions")
    print(f"Embedder: {metrics['embedder']}   Generator: {metrics['generator']}")
    print("-" * 62)
    print(f"Faithfulness (delivered claims)   {metrics['faithfulness_pct']:>6}%  "
          f"({metrics['claims_faithful']}/{metrics['claims_delivered']} claims)")
    print(f"Balance coverage                  {metrics['balance_coverage_pct']:>6}%  "
          f"({totals['balance_ok']}/{totals['balance_items']} contested questions)")
    print(f"Refusal correctness               {metrics['refusal_correct_pct']:>6}%  "
          f"({totals['refusal_ok']}/{totals['refusal_items']} refusal questions)")
    print(f"False refusals                    {metrics['false_refusal_rate_pct']:>6}%  "
          f"of {totals['nonrefusal_items']} answerable questions")
    print(f"Per-candidate status accuracy     {metrics['status_accuracy_pct']:>6}%  "
          f"({totals['status_ok']}/{totals['status_checks']} checks)")
    print(f"Retrieval hit-rate (proxy)        {metrics['retrieval_hit_rate_pct']:>6}%  "
          f"({totals['hit_ok']}/{totals['hit_items']} questions)")
    print(f"Critic removed                    {totals['critic_flagged']} of "
          f"{totals['critic_checked']} generated claims")
    failed = [e for e in per_item if e["problems"]]
    if failed:
        print("\nItems with problems:")
        for e in failed:
            print(f"  {e['id']}: {'; '.join(e['problems'])}")

    report = {"metrics": metrics, "items": per_item}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report: {args.report}")


if __name__ == "__main__":
    main()
