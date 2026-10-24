"""Answer generation, behind a provider interface.

Two paths:

* **LLM path** — used when ``ANTHROPIC_API_KEY`` (preferred) or
  ``OPENAI_API_KEY`` is present in the environment. The model receives the
  retrieved chunks and must answer in the JSON shape defined in
  :mod:`comparer.prompts`, citing chunk ids.
* **Extractive fallback (the no-key path)** — composes the answer purely
  from the retrieved chunks: the most question-relevant sentences from
  each candidate's top chunks, each presented as a claim followed by its
  citation. It cannot synthesise, compare nuance, or write a difference
  summary (that stays ``null``) — but every sentence it emits is verbatim
  source text with a real citation, so it is honest by construction.
  This is the path that runs with no API key configured.
"""

from __future__ import annotations

import json
import os
import re

from . import prompts
from .chunking import split_sentences
from .embeddings import tokenize
from .retrieval import RetrievedChunk

MAX_EXTRACTIVE_CLAIMS = 4


class Claim:
    def __init__(self, text: str, citations: list[str]) -> None:
        self.text = text
        self.citations = citations

    def to_dict(self) -> dict:
        return {"text": self.text, "citations": self.citations}


class CandidateAnswer:
    def __init__(self, status: str, claims: list[Claim]) -> None:
        self.status = status  # "answered" | "not_addressed"
        self.claims = claims

    def to_dict(self) -> dict:
        return {"status": self.status, "claims": [c.to_dict() for c in self.claims]}


class GeneratedAnswer:
    def __init__(
        self,
        per_candidate: dict[str, CandidateAnswer],
        difference_summary: str | None,
    ) -> None:
        self.per_candidate = per_candidate
        self.difference_summary = difference_summary

    def to_dict(self) -> dict:
        return {
            "per_candidate": {k: v.to_dict() for k, v in self.per_candidate.items()},
            "difference_summary": self.difference_summary,
        }


class ExtractiveGenerator:
    """No-API-key generator: verbatim sentences + citations, nothing else."""

    name = "extractive"

    def generate(
        self, question: str, grouped: dict[str, list[RetrievedChunk]]
    ) -> GeneratedAnswer:
        q_tokens = set(tokenize(question))
        per_candidate: dict[str, CandidateAnswer] = {}
        for candidate, results in grouped.items():
            if not results:
                per_candidate[candidate] = CandidateAnswer("not_addressed", [])
                continue
            scored: list[tuple[float, int, Claim]] = []
            order = 0
            for r in results:
                for sentence in split_sentences(r.chunk.text):
                    order += 1
                    if len(sentence) < 25 or sentence.lower().startswith("read more"):
                        continue
                    overlap = len(q_tokens & set(tokenize(sentence)))
                    scored.append((overlap, order, Claim(sentence, [r.chunk.id])))
            if not scored:
                per_candidate[candidate] = CandidateAnswer("not_addressed", [])
                continue
            # Prefer sentences overlapping the question; break ties by
            # source order so the answer reads in document order.
            scored.sort(key=lambda item: (-item[0], item[1]))
            chosen = sorted(scored[:MAX_EXTRACTIVE_CLAIMS], key=lambda item: item[1])
            per_candidate[candidate] = CandidateAnswer(
                "answered", [claim for _, _, claim in chosen]
            )
        # No synthesis without an LLM: a difference summary written by
        # stitching source sentences together would misrepresent both
        # candidates, so it stays null on this path.
        return GeneratedAnswer(per_candidate, difference_summary=None)


class LLMGenerator:
    """LLM-backed generator (Anthropic preferred, OpenAI otherwise)."""

    name = "llm"

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.model = os.environ.get(
            "LLM_MODEL",
            "claude-sonnet-4-5" if provider == "anthropic" else "gpt-4o-mini",
        )

    def generate(
        self, question: str, grouped: dict[str, list[RetrievedChunk]]
    ) -> GeneratedAnswer:
        messages = prompts.build_messages(question, grouped)
        if self.provider == "anthropic":
            raw = self._call_anthropic(messages)
        else:
            raw = self._call_openai(messages)
        return self._parse(raw, grouped)

    def _call_anthropic(self, messages: list[dict]) -> str:
        import anthropic

        client = anthropic.Anthropic()
        system = messages[0]["content"]
        resp = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": messages[1]["content"]}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def _call_openai(self, messages: list[dict]) -> str:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def _parse(self, raw: str, grouped: dict[str, list[RetrievedChunk]]) -> GeneratedAnswer:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("LLM did not return a JSON object")
        data = json.loads(match.group(0))
        per_candidate: dict[str, CandidateAnswer] = {}
        for candidate in grouped:
            entry = data.get("per_candidate", {}).get(candidate, {})
            claims = [
                Claim(str(c.get("text", "")), [str(x) for x in c.get("citations", [])])
                for c in entry.get("claims", [])
            ]
            status = entry.get("status", "answered" if claims else "not_addressed")
            per_candidate[candidate] = CandidateAnswer(status, claims)
        summary = data.get("difference_summary")
        return GeneratedAnswer(per_candidate, summary if isinstance(summary, str) else None)


def get_generator():
    """Pick the generator based on the environment. Anthropic key wins;
    OpenAI key next; otherwise the extractive no-key path."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMGenerator("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        return LLMGenerator("openai")
    return ExtractiveGenerator()
