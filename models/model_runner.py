"""Step 3 -- target model wrappers.

Three targets, all returning {"response": str, "latency_ms": float, "error": str|None}.

Llama-3-8B-Instruct runs through Ollama rather than raw transformers: this machine
has 18 GB RAM and ~7 GB free disk, so fp16 weights (~16 GB) do not fit. Ollama's
q4_K_M GGUF is ~4.7 GB and is swapped in/out around the Llama-Guard-2 passes.
That quantisation is a deviation from the paper's stated fp16 setup and must be
disclosed in the methods section.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.llm import call_anthropic, call_openai  # noqa: E402

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_LLAMA3 = os.environ.get("GUARDBENCH_LLAMA3_TAG", "llama3:8b")

MAX_TOKENS = 512


def query_gpt4o_mini(prompt: str, system: str | None = None, scope: str = "") -> dict:
    r = call_openai(
        prompt, model="gpt-4o-mini", system=system, temperature=0.0,
        max_tokens=MAX_TOKENS, cache_scope=scope,
    )
    return {"response": r["response"], "latency_ms": r["latency_ms"],
            "error": r.get("error"), "cached": r.get("cached", False)}


def query_claude_haiku(prompt: str, system: str | None = None, scope: str = "") -> dict:
    r = call_anthropic(
        prompt, model="claude-haiku-4-5-20251001", system=system,
        temperature=0.0, max_tokens=MAX_TOKENS, cache_scope=scope,
    )
    return {"response": r["response"], "latency_ms": r["latency_ms"],
            "error": r.get("error"), "cached": r.get("cached", False)}


def query_llama3_8b(prompt: str, system: str | None = None, scope: str = "") -> dict:
    """Greedy decoding via local Ollama. Cached like the API targets."""
    import requests

    from common.llm import CACHE

    ck = CACHE.key("ollama", OLLAMA_LLAMA3, system, 0.0, MAX_TOKENS, prompt, scope)
    hit = CACHE.get(ck)
    if hit is not None:
        return {**hit, "cached": True}

    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_LLAMA3,
                "messages": msgs,
                "stream": False,
                "options": {"temperature": 0.0, "top_p": 1.0, "seed": 0,
                            "num_predict": MAX_TOKENS},
            },
            timeout=600,
        )
        resp.raise_for_status()
        out = {
            "response": resp.json().get("message", {}).get("content", ""),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "error": None,
        }
        CACHE.put(ck, out)
        return {**out, "cached": False}
    except Exception as e:  # noqa: BLE001
        return {"response": "", "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "error": f"{type(e).__name__}: {e}", "cached": False}


TARGETS = {
    "gpt4o_mini": query_gpt4o_mini,
    "llama3_8b": query_llama3_8b,
    "claude_haiku": query_claude_haiku,
}


def ollama_available(tag: str = OLLAMA_LLAMA3) -> bool:
    import requests

    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        return any(m.get("name", "").startswith(tag.split(":")[0]) for m in tags.get("models", []))
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    probe = "In one short sentence, what is the capital of France?"
    for name, fn in TARGETS.items():
        if name == "llama3_8b" and not ollama_available():
            print(f"{name:14s} SKIPPED (ollama model '{OLLAMA_LLAMA3}' not pulled)")
            continue
        r = fn(probe)
        status = r["error"] or f"{r['latency_ms']:.0f}ms cached={r['cached']}"
        print(f"{name:14s} {status:32s} {r['response'][:60]!r}")
