"""Shared LLM plumbing for GuardBench.

Provides:
  * TokenBucket / RateLimiter  -- respects the *measured* account limits, not guesses.
  * SqliteCache                -- content-addressed response cache so multi-hour runs resume free.
  * call_openai / call_anthropic -- retrying, cached, thread-safe callers.

Measured limits on this account (from response headers, 2026-08-15):
  OpenAI   gpt-4o-mini : 10,000 RPM / 200,000 TPM   (TPM is binding)
  Anthropic haiku-4.5  : 10,000 RPM / 12,000,000 TPM
We run comfortably under these so a co-running job cannot push us into sustained 429s.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from typing import Any

# ----------------------------------------------------------------------------- limits

# Fractions of the measured ceilings (OpenAI 10,000 RPM / 200,000 TPM;
# Anthropic 10,000 RPM / 12,000,000 TPM). The earlier 300/400 RPM settings were
# far below the account limits and became the binding constraint themselves,
# capping throughput at ~5 req/s regardless of latency. TPM is the real ceiling,
# so RPM is now set high enough to stay out of the way.
OPENAI_RPM = 2_000
OPENAI_TPM = 180_000
ANTHROPIC_RPM = 2_000
ANTHROPIC_TPM = 2_000_000

MAX_RETRIES = 8


class TokenBucket:
    """Classic token bucket. `capacity` units refill over 60s."""

    def __init__(self, capacity: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.rate = capacity / 60.0  # units per second
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def take(self, n: float = 1.0) -> None:
        n = min(n, self.capacity)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
                self.ts = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                deficit = n - self.tokens
                wait = deficit / self.rate
            time.sleep(min(wait, 5.0))


class RateLimiter:
    def __init__(self, rpm: float, tpm: float):
        self.req = TokenBucket(rpm)
        self.tok = TokenBucket(tpm)

    def acquire(self, est_tokens: int) -> None:
        self.req.take(1)
        self.tok.take(max(1, est_tokens))


OPENAI_LIMITER = RateLimiter(OPENAI_RPM, OPENAI_TPM)
ANTHROPIC_LIMITER = RateLimiter(ANTHROPIC_RPM, ANTHROPIC_TPM)


def est_tokens(text: str, max_out: int) -> int:
    """Cheap char/4 heuristic; only needs to be right to ~30% for limiter purposes."""
    return len(text) // 4 + max_out + 16


# ----------------------------------------------------------------------------- cache


class SqliteCache:
    """Content-addressed cache. Key = sha256 of (provider, model, params, prompt)."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.commit()
        self.lock = threading.Lock()

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k: str):
        with self.lock:
            row = self.conn.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: Any) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v))
            )
            self.conn.commit()


_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llm_cache.sqlite")
CACHE = SqliteCache(_CACHE_PATH)


# ----------------------------------------------------------------------------- clients

_openai_client = None
_anthropic_client = None
_client_lock = threading.Lock()


def _openai():
    global _openai_client
    with _client_lock:
        if _openai_client is None:
            from openai import OpenAI

            _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
    return _openai_client


def _anthropic():
    global _anthropic_client
    with _client_lock:
        if _anthropic_client is None:
            import anthropic

            _anthropic_client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0
            )
    return _anthropic_client


def _retry_sleep(attempt: int, err: Exception) -> float:
    retry_after = getattr(getattr(err, "response", None), "headers", {}) or {}
    try:
        ra = float(retry_after.get("retry-after", 0))
        if ra > 0:
            return min(ra, 60.0)
    except Exception:
        pass
    return min(60.0, (2 ** attempt) * 0.5 * (1.0 + random.random()))


# --- daily-quota (RPD) gate -------------------------------------------------
# The OpenAI org has a 10,000 requests/day cap. A per-minute 429 is worth an
# exponential backoff; a per-DAY 429 is not -- retrying it just burns the retry
# budget and returns errors. Instead we park all threads until the quota has had
# time to replenish, so a long run degrades to the replenishment rate and
# eventually completes rather than failing.
_RPD_UNTIL = 0.0
_RPD_LOCK = threading.Lock()
RPD_PARK_SECONDS = 300.0


def _is_rpd_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "requests per day" in msg or "rpd" in msg


def _rpd_park() -> None:
    global _RPD_UNTIL
    with _RPD_LOCK:
        _RPD_UNTIL = max(_RPD_UNTIL, time.time() + RPD_PARK_SECONDS)


def _rpd_wait() -> None:
    while True:
        with _RPD_LOCK:
            until = _RPD_UNTIL
        remaining = until - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30.0))


def _is_retryable(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    name = type(err).__name__
    if status is not None:
        return status == 429 or status >= 500
    return any(s in name for s in ("RateLimit", "APIConnection", "Timeout", "InternalServer", "Overloaded", "APIStatus"))


def call_openai(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_cache: bool = True,
    cache_scope: str = "",
) -> dict:
    """Returns {"response": str, "latency_ms": float, "cached": bool, "error": str|None}.

    `cache_scope` partitions the cache. run_experiments.py passes the run index so
    replicate runs issue genuinely fresh calls -- otherwise temperature=0 caching
    would make runs 2 and 3 byte-identical and pooling n=4200x3 for CIs would claim
    a sqrt(3) precision gain from zero additional information.
    """
    ck = CACHE.key("openai", model, system, temperature, max_tokens, prompt, cache_scope)
    if use_cache:
        hit = CACHE.get(ck)
        if hit is not None:
            return {**hit, "cached": True}

    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    last = None
    attempt = 0
    while attempt < MAX_RETRIES:
        _rpd_wait()
        OPENAI_LIMITER.acquire(est_tokens(prompt + (system or ""), max_tokens))
        t0 = time.perf_counter()
        try:
            r = _openai().chat.completions.create(
                model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens
            )
            out = {
                "response": r.choices[0].message.content or "",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "error": None,
            }
            if use_cache:
                CACHE.put(ck, out)
            return {**out, "cached": False}
        except Exception as e:  # noqa: BLE001
            last = e
            if _is_rpd_error(e):
                # Daily cap: park everyone, do NOT consume a retry slot.
                _rpd_park()
                continue
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                break
            time.sleep(_retry_sleep(attempt, e))
            attempt += 1
    return {"response": "", "latency_ms": 0.0, "cached": False, "error": f"{type(last).__name__}: {last}"}


def call_anthropic(
    prompt: str,
    *,
    model: str = "claude-haiku-4-5-20251001",
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_cache: bool = True,
    cache_scope: str = "",
) -> dict:
    ck = CACHE.key("anthropic", model, system, temperature, max_tokens, prompt, cache_scope)
    if use_cache:
        hit = CACHE.get(ck)
        if hit is not None:
            return {**hit, "cached": True}

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    last = None
    for attempt in range(MAX_RETRIES):
        ANTHROPIC_LIMITER.acquire(est_tokens(prompt + (system or ""), max_tokens))
        t0 = time.perf_counter()
        try:
            r = _anthropic().messages.create(**kwargs)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            out = {
                "response": text,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "error": None,
            }
            if use_cache:
                CACHE.put(ck, out)
            return {**out, "cached": False}
        except Exception as e:  # noqa: BLE001
            last = e
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                break
            time.sleep(_retry_sleep(attempt, e))
    return {"response": "", "latency_ms": 0.0, "cached": False, "error": f"{type(last).__name__}: {last}"}
