#!/usr/bin/env python3
"""Step 7 -- rewrite the paper's numeric tables from measured results.

Operates on labelled table blocks in paper1_MDPI.tex, replacing only the data
rows between \\midrule and \\bottomrule so surrounding formatting, captions and
column specs are untouched.

    python3 update_paper.py --dry-run          # show the diff, write nothing
    python3 update_paper.py --tables results   # rewrite Table 3 only
    python3 update_paper.py --tables all

Deliberately does NOT touch prose. Narrative claims ("ordering is consistent
across models") are re-checked by hand against results/metrics.json, because a
regex that rewrites sentences is far more likely to introduce a false claim than
to fix one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(ROOT, "paper1_MDPI.tex")
METRICS = os.path.join(ROOT, "results", "metrics.json")

GUARD_LABEL = {
    "G1": "G1 Rule  ", "G2": "G2 Class.", "G3": "G3 LG2   ",
    "G4": "G4 Smooth", "none": r"\textit{No Guard}",
}
MODEL_ORDER = ["gpt4o_mini", "llama3_8b", "claude_haiku"]


def fnum(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def latex_ms(ms: float) -> str:
    if ms is None or ms != ms:
        return "--"
    v = int(round(ms))
    s = f"{v:,}".replace(",", "{,}")
    return s


def get_table_block(tex: str, label: str) -> tuple[int, int]:
    """Return (start, end) offsets of the data-row region of a labelled table."""
    m = re.search(r"\\begin\{table\}.*?\\label\{" + re.escape(label) + r"\}.*?\\end\{table\}",
                  tex, re.S)
    if not m:
        raise KeyError(f"table {label} not found")
    block = m.group(0)
    # data rows sit between the FIRST \midrule and the \bottomrule
    mid = block.index(r"\midrule") + len(r"\midrule")
    bot = block.index(r"\bottomrule")
    return m.start() + mid, m.start() + bot


def build_results_table(cells: dict, model: str = "gpt4o_mini") -> str:
    """Table 3: overall + GCG ASR/CSS, FPR, dt95 for each guardrail."""
    rows = []
    for g in ["G1", "G2", "G3", "G4", "none"]:
        c = cells.get(f"{g}|{model}")
        if not c:
            continue
        gcg = c["by_category"].get("Gradient-Optimised Suffix", {})
        is_none = g == "none"
        css_o = "--" if is_none else fnum(c["css"], 2)
        css_g = "--" if is_none else fnum(gcg.get("css"), 2)
        fpr_s = ("0.0" if is_none else
                 f"${fnum(c['fpr']*100 if c['fpr'] is not None else None)}"
                 f"{{\\pm}}{fnum(c['fpr_ci_pp'])}$")
        rows.append(
            f"    {GUARD_LABEL[g]} & "
            f"${fnum(c['asr']*100)}{{\\pm}}{fnum(c['asr_ci_pp'])}$ & {css_o} & "
            f"${fnum(gcg.get('asr', float('nan'))*100 if gcg else None)}"
            f"{{\\pm}}{fnum(gcg.get('asr_ci_pp'))}$ & {css_g} & "
            f"{fpr_s} & {latex_ms(c['dt95_ms'])} \\\\"
        )
        if g == "G4":
            rows.append(r"    \midrule")
    return "\n" + "\n".join(rows) + "\n    "


def build_multimodel_table(cells: dict) -> str:
    """Table 4: overall ASR per guardrail x target model."""
    rows = []
    names = {"G1": "G1 Rule-Based ", "G2": "G2 Classifier ",
             "G3": "G3 LG2        ", "G4": "G4 SmoothLLM  ", "none": r"\textit{No Guard}"}
    for g in ["G1", "G2", "G3", "G4", "none"]:
        parts = []
        any_cell = False
        for m in MODEL_ORDER:
            c = cells.get(f"{g}|{m}")
            if c:
                any_cell = True
                parts.append(f"${fnum(c['asr']*100)}{{\\pm}}{fnum(c['asr_ci_pp'])}$")
            else:
                parts.append("--")
        if any_cell:
            rows.append(f"    {names[g]} & " + " & ".join(parts) + r" \\")
    return "\n" + "\n".join(rows) + "\n    "


def apply(tex: str, label: str, body: str) -> str:
    s, e = get_table_block(tex, label)
    return tex[:s] + body + tex[e:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="+", default=["all"],
                    choices=["all", "results", "multimodel"])
    ap.add_argument("--variant", default="excl_copyright",
                    choices=["incl_copyright", "excl_copyright"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    metrics = json.load(open(METRICS))
    cells = metrics[args.variant]["cells"]
    tex = open(TEX).read()
    orig = tex

    want = set(args.tables)
    if "all" in want:
        want = {"results", "multimodel"}

    if "results" in want:
        tex = apply(tex, "tab:results", build_results_table(cells))
    if "multimodel" in want:
        tex = apply(tex, "tab:multimodel", build_multimodel_table(cells))

    if args.dry_run:
        import difflib

        for line in difflib.unified_diff(orig.splitlines(), tex.splitlines(),
                                         "before", "after", lineterm="", n=2):
            print(line)
        return

    shutil.copy(TEX, TEX + ".bak")
    open(TEX, "w").write(tex)
    print(f"updated {TEX} (backup at {TEX}.bak)")


if __name__ == "__main__":
    main()
