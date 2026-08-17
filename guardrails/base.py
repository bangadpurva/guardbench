"""Common guardrail interface for GuardBench.

Every guardrail returns the same dict shape so run_experiments.py is agnostic:

    {"blocked": bool, "latency_ms": float, "raw_score": float | None, "detail": dict}

Latency contract: `latency_ms` measures only the work a deployment would pay for
at request time (matching, tokenisation, forward pass). One-off model/regex
loading happens in __init__ and is excluded, otherwise the first prompt would
dominate the p95.
"""
from __future__ import annotations

import time
from typing import Any


class GuardrailBase:
    name = "base"

    # A guardrail verdict is a function of the prompt alone -- it never sees the
    # target model. Re-evaluating an 8B local classifier once per target model
    # would triple hours of GPU time to recompute identical answers, so
    # deterministic model-backed guardrails set cacheable=True and the verdict
    # (including its first *honestly measured* latency) is replayed across cells.
    # Latency is therefore measured once per distinct prompt, not once per cell.
    cacheable = False
    cache_version = "v1"

    def warmup(self) -> None:
        """Optional: run a throwaway inference so lazy init doesn't skew latency."""

    def _cache_key(self, prompt: str) -> str:
        from common.llm import CACHE

        return CACHE.key("guardrail", self.name, self.cache_version, prompt)

    def _evaluate(self, prompt: str) -> tuple[bool, float | None, dict]:
        raise NotImplementedError

    def evaluate(self, prompt: str) -> dict[str, Any]:
        if self.cacheable:
            from common.llm import CACHE

            ck = self._cache_key(prompt)
            hit = CACHE.get(ck)
            if hit is not None:
                return {**hit, "cached": True}

        t0 = time.perf_counter()
        blocked, score, detail = self._evaluate(prompt)
        dt = (time.perf_counter() - t0) * 1000.0
        out = {
            "blocked": bool(blocked),
            "latency_ms": dt,
            "raw_score": score,
            "detail": detail,
        }
        if self.cacheable:
            from common.llm import CACHE

            CACHE.put(self._cache_key(prompt), out)
        return {**out, "cached": False}
