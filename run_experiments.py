#!/usr/bin/env python3
"""Step 5 -- experiment runner.

For each (run, guardrail, target model):
  attack prompts -> guardrail.evaluate -> (if not blocked) target -> adjudicate
  benign prompts -> guardrail.evaluate -> FPR event

Design notes
------------
* Resumable. Every completed (run, guardrail, model, prompt_id) key is read back
  from the JSONL on start and skipped, so an interrupted multi-hour run resumes
  where it stopped without re-spending API budget.
* Guardrail evaluation is serialised per worker but target/judge calls are
  concurrent, bounded by the token-bucket limiter in common/llm.py.
* Latency: `guardrail_latency_ms` is wall-clock for the guardrail only. Local
  guardrails (G2/G3/G4) are run on a single worker thread so their measured
  latency reflects real serving cost rather than contention between threads.

Usage:
    python3 run_experiments.py --guardrails G1 --models gpt4o_mini --runs 1 --limit 50
    python3 run_experiments.py --guardrails G1 G2 G3 G4 --runs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adjudicator.judge import adjudicate  # noqa: E402
from models.model_runner import TARGETS, ollama_available  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(ROOT, "results", "raw")
# Guardrail-only rows live in a separate tree so they never satisfy the resume
# key of a full judged run (which must still issue target + judge calls).
RAW_GUARDONLY = os.path.join(ROOT, "results", "raw_guardonly")

# Guardrails that hold a model in memory: run single-threaded for honest latency.
LOCAL_GUARDRAILS = {"G2", "G3", "G4"}

_write_lock = threading.Lock()

# Circuit breaker: if the backend starts failing (expired credits, revoked key,
# sustained 5xx), stop fast and loudly instead of grinding through thousands of
# prompts producing unusable rows.
ERRORS: list[str] = []
GUARDRAIL_ONLY = False
ERROR_ABORT_THRESHOLD = 50


class BackendFailure(RuntimeError):
    pass


def load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build_guardrail(name: str, seed: int = 0):
    if name == "G1":
        from guardrails.g1_rules import G1RuleFilter

        return G1RuleFilter()
    if name == "G2":
        from guardrails.g2_classifier import G2Classifier

        return G2Classifier()
    if name == "G3":
        from guardrails.g3_llamaguard import G3LlamaGuard

        return G3LlamaGuard()
    if name == "G4":
        from guardrails.g4_smoothllm import G4SmoothLLM

        return G4SmoothLLM(seed=seed)
    if name == "none":
        from guardrails.g0_none import G0NoGuard

        return G0NoGuard()
    raise ValueError(f"unknown guardrail {name}")


def done_keys(path: str) -> set[tuple]:
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
                keys.add((r["run"], r["guardrail"], r["model"], r["prompt_id"], r["split"]))
            except Exception:  # noqa: BLE001, tolerate a torn final line
                continue
    return keys


def append_jsonl(path: str, row: dict) -> None:
    with _write_lock:
        with open(path, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()


def run_cell(guard_name: str, model_name: str, run_idx: int, attacks: list[dict],
             benign: list[dict], workers: int) -> None:
    """One (guardrail, model, run) cell of the experiment matrix."""
    base = RAW_GUARDONLY if GUARDRAIL_ONLY else RAW
    out_path = os.path.join(base, f"{guard_name}_{model_name}.jsonl")
    os.makedirs(base, exist_ok=True)
    already = done_keys(out_path)

    guard = build_guardrail(guard_name, seed=run_idx)
    guard.warmup()
    target = TARGETS[model_name]
    scope = f"run{run_idx}"

    todo_a = [p for p in attacks
              if (run_idx, guard_name, model_name, p["id"], "attack") not in already]
    todo_b = [p for p in benign
              if (run_idx, guard_name, model_name, p["id"], "benign") not in already]

    if not todo_a and not todo_b:
        print(f"  [skip] {guard_name}/{model_name}/run{run_idx} already complete")
        return

    # Local model guardrails must not be hammered from many threads or the measured
    # latency becomes a contention artefact rather than a serving cost.
    guard_lock = threading.Lock() if guard_name in LOCAL_GUARDRAILS else None
    eff_workers = 1 if guard_name in LOCAL_GUARDRAILS else workers

    def guard_eval(text: str) -> dict:
        if guard_lock:
            with guard_lock:
                return guard.evaluate(text)
        return guard.evaluate(text)

    def do_attack(p: dict) -> dict | None:
        g = guard_eval(p["text"])
        if GUARDRAIL_ONLY:
            # Guardrail-side metrics only (block rate, FPR, latency). No target and
            # no adjudication, so this runs with zero API quota. classification is
            # recorded as "unjudged" and compute_metrics excludes those rows from
            # ASR rather than counting them as safe.
            return {
                "run": run_idx, "guardrail": guard_name, "model": model_name,
                "split": "attack", "prompt_id": p["id"], "category": p["category"],
                "severity": p.get("severity"), "is_copyright": p.get("is_copyright", False),
                "template_id": p.get("template_id"),
                "blocked": g["blocked"], "guardrail_latency_ms": g["latency_ms"],
                "raw_score": g["raw_score"], "model_latency_ms": 0.0,
                "classification": "unjudged", "model_error": None, "response_chars": 0,
            }
        if g["blocked"]:
            resp, mlat, merr = "[BLOCKED]", 0.0, None
        else:
            m = target(p["text"], scope=scope)
            resp, mlat, merr = m["response"], m["latency_ms"], m.get("error")
        if merr:
            # A failed target call yields an empty response, which the adjudicator
            # would score "safe" -- silently biasing ASR downward. Drop the row so
            # the resume logic retries it, and let the circuit breaker decide
            # whether the run should stop. (Learned the hard way: an exhausted
            # credit balance produced 1,315 such rows before it was caught.)
            ERRORS.append(merr)
            return None
        cls = adjudicate(p["text"], resp, scope=scope)
        return {
            "run": run_idx, "guardrail": guard_name, "model": model_name,
            "split": "attack", "prompt_id": p["id"], "category": p["category"],
            "severity": p.get("severity"), "is_copyright": p.get("is_copyright", False),
            "template_id": p.get("template_id"),
            "blocked": g["blocked"], "guardrail_latency_ms": g["latency_ms"],
            "raw_score": g["raw_score"], "model_latency_ms": mlat,
            "classification": cls, "model_error": merr,
            "response_chars": len(resp or ""),
        }

    def do_benign(p: dict) -> dict:
        g = guard_eval(p["text"])
        return {
            "run": run_idx, "guardrail": guard_name, "model": model_name,
            "split": "benign", "prompt_id": p["id"], "category": "benign",
            "severity": 0, "is_copyright": False, "template_id": None,
            "blocked": g["blocked"], "guardrail_latency_ms": g["latency_ms"],
            "raw_score": g["raw_score"], "model_latency_ms": 0.0,
            "classification": "n/a", "model_error": None, "response_chars": 0,
        }

    desc = f"{guard_name}/{model_name}/run{run_idx}"
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=eff_workers) as ex:
        futs = [ex.submit(do_attack, p) for p in todo_a]
        futs += [ex.submit(do_benign, p) for p in todo_b]
        n_written = 0
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc, unit="p"):
            row = f.result()
            if row is not None:
                append_jsonl(out_path, row)
                n_written += 1
            if len(ERRORS) >= ERROR_ABORT_THRESHOLD:
                for fut in futs:
                    fut.cancel()
                raise BackendFailure(
                    f"{len(ERRORS)} target-model errors in {desc}; aborting.\n"
                    f"  most recent: {ERRORS[-1][:200]}\n"
                    f"  {n_written} good rows were written and are safe to keep."
                )
    print(f"  [done] {desc} in {time.time()-t0:.0f}s "
          f"({n_written} rows written; {len(ERRORS)} errors so far)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guardrails", nargs="+", default=["G1"])
    ap.add_argument("--models", nargs="+", default=["gpt4o_mini"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--run-offset", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="use only first N attack/benign prompts")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-benign", action="store_true")
    ap.add_argument("--guardrail-only", action="store_true",
                    help="score guardrail block/FPR/latency only; no target or judge calls")
    args = ap.parse_args()

    global GUARDRAIL_ONLY
    GUARDRAIL_ONLY = args.guardrail_only

    attacks = load_jsonl(os.path.join(DATA, "attack_prompts.jsonl"))
    benign = [] if args.no_benign else load_jsonl(os.path.join(DATA, "benign_prompts.jsonl"))
    if args.limit:
        # Stratify by category: prompts are composed category-by-category, so a
        # plain head-slice would sample a single category and give a meaningless
        # pilot ASR.
        import collections
        import random as _random

        by_cat: dict[str, list[dict]] = collections.defaultdict(list)
        for p in attacks:
            by_cat[p["category"]].append(p)
        per = max(1, args.limit // len(by_cat))
        rs = _random.Random(1234)
        picked: list[dict] = []
        for cat in sorted(by_cat):
            picked.extend(rs.sample(by_cat[cat], min(per, len(by_cat[cat]))))
        attacks = sorted(picked, key=lambda p: p["id"])
        benign = rs.sample(benign, min(max(1, args.limit // 4), len(benign))) if benign else []

    if "llama3_8b" in args.models and not ollama_available():
        print("ERROR: llama3_8b requested but the Ollama model is not pulled.",
              file=sys.stderr)
        sys.exit(2)

    print(f"attacks={len(attacks)} benign={len(benign)} "
          f"guardrails={args.guardrails} models={args.models} runs={args.runs}")

    for run_idx in range(args.run_offset, args.run_offset + args.runs):
        for guard_name in args.guardrails:
            for model_name in args.models:
                run_cell(guard_name, model_name, run_idx, attacks, benign, args.workers)


if __name__ == "__main__":
    main()
