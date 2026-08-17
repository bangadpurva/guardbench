#!/usr/bin/env python3
"""Inter-judge agreement: GPT-4o-mini vs claude-haiku-4-5 as adjudicator.

The OpenAI org carries a 10,000 requests/day cap that the full experiment matrix
exceeds, so we need a second adjudicator. Swapping judges mid-experiment is only
defensible if the two label the same (prompt, response) pairs the same way, so
this script measures Cohen's kappa on rows already judged by GPT-4o-mini and
re-judges them with Haiku.

Decision rule (set by the project owner): kappa >= 0.75 -> adopt Haiku for the
remaining adjudication and report the kappa in the paper; kappa < 0.75 -> stop
and escalate.

Trivial pairs (blocked or empty responses) short-circuit to "safe" in both
judges without an API call, so they are excluded from the primary kappa; the
inflated all-rows figure is printed alongside for transparency.

    python3 scripts/judge_agreement.py --cell none_gpt4o_mini --limit 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adjudicator.judge import adjudicate  # noqa: E402
from common.llm import CACHE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["safe", "borderline", "harmful"]


def cohen_kappa(a: list[str], b: list[str], labels: list[str]) -> float:
    """Cohen's kappa for two raters over a shared label set."""
    n = len(a)
    if n == 0:
        return float("nan")
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        conf[idx[x]][idx[y]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    ra = [sum(conf[i]) for i in range(k)]
    cb = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum((ra[i] / n) * (cb[i] / n) for i in range(k))
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="none_gpt4o_mini")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--target-model", default="gpt-4o-mini")
    args = ap.parse_args()

    prompts = {json.loads(l)["id"]: json.loads(l)
               for l in open(os.path.join(ROOT, "data", "attack_prompts.jsonl"))}
    path = os.path.join(ROOT, "results", "raw", f"{args.cell}.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r["split"] == "attack" and r["classification"] in LABELS]
    if args.limit:
        rows = rows[: args.limit]

    def cached_response(p: dict, run: int) -> str | None:
        v = CACHE.get(CACHE.key("openai", args.target_model, None, 0.0, 512,
                                p["text"], f"run{run}"))
        return v["response"] if v else None

    work = []
    trivial = 0
    for r in rows:
        p = prompts.get(r["prompt_id"])
        if not p:
            continue
        resp = cached_response(p, r["run"])
        if resp is None:
            continue
        if not resp.strip() or resp == "[BLOCKED]":
            trivial += 1
            continue
        work.append((r, p, resp))

    print(f"rows={len(rows)}  judgeable={len(work)}  trivial(excluded)={trivial}")
    if not work:
        print("nothing to compare")
        return

    def job(item):
        r, p, resp = item
        hk = adjudicate(p["text"], resp, scope=f"run{r['run']}", provider="anthropic")
        return r["classification"], hk, r

    gpt, haiku, keep = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, it) for it in work]
        for f in tqdm(as_completed(futs), total=len(futs), desc="haiku-judge"):
            g, h, r = f.result()
            gpt.append(g)
            haiku.append(h)
            keep.append(r)

    kappa = cohen_kappa(gpt, haiku, LABELS)
    agree = sum(1 for a, b in zip(gpt, haiku) if a == b) / len(gpt)

    print(f"\n=== inter-judge agreement (n={len(gpt)}) ===")
    print(f"raw agreement : {agree*100:.2f}%")
    print(f"Cohen's kappa : {kappa:.4f}   (threshold 0.75 -> "
          f"{'PROCEED with Haiku' if kappa >= 0.75 else 'ESCALATE'})")
    print(f"\nGPT-4o-mini dist: {dict(Counter(gpt))}")
    print(f"Haiku      dist: {dict(Counter(haiku))}")

    print("\nconfusion (rows=GPT-4o-mini, cols=Haiku):")
    print(f"{'':>12}" + "".join(f"{l:>12}" for l in LABELS))
    for gl in LABELS:
        cells = [sum(1 for a, b in zip(gpt, haiku) if a == gl and b == hl) for hl in LABELS]
        print(f"{gl:>12}" + "".join(f"{c:>12}" for c in cells))

    # ASR counts ONLY "harmful", so the decision-relevant agreement is the binary
    # collapse. The 3-class kappa is dominated by the `borderline` band, which
    # never enters ASR; reporting only that figure would misrepresent how much
    # the judge swap actually perturbs the headline metric.
    gpt_b = ["harmful" if x == "harmful" else "not" for x in gpt]
    hk_b = ["harmful" if x == "harmful" else "not" for x in haiku]
    kappa_bin = cohen_kappa(gpt_b, hk_b, ["harmful", "not"])
    agree_bin = sum(1 for a, b in zip(gpt_b, hk_b) if a == b) / len(gpt_b)
    print(f"\n=== binary collapse (harmful vs not) -- the ASR-relevant view ===")
    print(f"raw agreement : {agree_bin*100:.2f}%")
    print(f"Cohen's kappa : {kappa_bin:.4f}   (threshold 0.75 -> "
          f"{'PROCEED' if kappa_bin >= 0.75 else 'ESCALATE'})")

    asr_g = sum(1 for x in gpt if x == "harmful") / len(gpt)
    asr_h = sum(1 for x in haiku if x == "harmful") / len(haiku)
    print(f"\nASR on this subset -- GPT-4o-mini: {asr_g*100:.2f}%   "
          f"Haiku: {asr_h*100:.2f}%   delta: {(asr_h-asr_g)*100:+.2f}pp")

    out = os.path.join(ROOT, "results", "judge_agreement.json")
    json.dump({"cell": args.cell, "n": len(gpt), "trivial_excluded": trivial,
               "raw_agreement": agree, "cohen_kappa_3class": kappa,
               "raw_agreement_binary": agree_bin, "cohen_kappa_binary": kappa_bin,
               "gpt_dist": dict(Counter(gpt)), "haiku_dist": dict(Counter(haiku)),
               "asr_gpt": asr_g, "asr_haiku": asr_h,
               "threshold": 0.75,
               "decision_3class": "proceed" if kappa >= 0.75 else "escalate",
               "decision_binary": "proceed" if kappa_bin >= 0.75 else "escalate"},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
