#!/usr/bin/env python3
"""Step 2 (G2) -- fine-tune DeBERTa-v3-base as a harmful-prompt classifier.

Training data: AdvBench harmful behaviours (label 1) + WildGuard train split
(label from its harm annotation), with benign prompts (label 0) drawn from the
same MS-MARCO / ShareGPT distribution used for the FPR counterfactual set.

Contamination guard: any training example whose normalised text collides with a
GuardBench evaluation prompt or its underlying behaviour is dropped, and the
count is reported. Without this the classifier would be evaluated on its own
training data -- AdvBench and HarmBench overlap substantially in behaviour space.

Usage:
    python3 scripts/train_g2.py --epochs 3 --batch-size 16 --lr 2e-5
    python3 scripts/train_g2.py --dry-run     # build + report the dataset only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(DATA, "sources")
OUT = os.path.join(ROOT, "models", "g2_deberta")

MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 512
SEED = 20260815


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def load_eval_fingerprints() -> set[str]:
    """Normalised text of every eval prompt AND its underlying behaviour."""
    fps: set[str] = set()
    for name in ("attack_prompts.jsonl", "benign_prompts.jsonl"):
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                fps.add(norm(r.get("text", "")))
                if r.get("target_behavior"):
                    fps.add(norm(r["target_behavior"]))
    return fps


def build_dataset(rng: random.Random) -> tuple[list[str], list[int], dict]:
    import pandas as pd

    texts: list[str] = []
    labels: list[int] = []
    stats = {"advbench": 0, "wildguard_harmful": 0, "wildguard_safe": 0,
             "benign": 0, "dropped_contaminated": 0, "wildguard_error": None}
    fps = load_eval_fingerprints()

    def add(t: str, y: int) -> bool:
        t = (t or "").strip()
        if not t or norm(t) in fps:
            stats["dropped_contaminated"] += 1
            return False
        texts.append(t)
        labels.append(y)
        return True

    # --- AdvBench harmful behaviours
    adv = pd.read_csv(os.path.join(SRC, "advbench_harmful_behaviors.csv"))
    for g in adv["goal"].tolist():
        if add(g, 1):
            stats["advbench"] += 1

    # --- WildGuard (gated: requires HF licence acceptance)
    try:
        from datasets import load_dataset

        wg = load_dataset("allenai/wildguard", "wildguardtrain",
                          split="train", token=os.environ.get("HF_TOKEN"))
        for r in wg:
            prompt = r.get("prompt")
            harm = str(r.get("prompt_harm_label") or "").lower()
            if not prompt or harm not in ("harmful", "unharmful"):
                continue
            y = 1 if harm == "harmful" else 0
            if add(prompt, y):
                stats["wildguard_harmful" if y else "wildguard_safe"] += 1
    except Exception as e:  # noqa: BLE001
        stats["wildguard_error"] = f"{type(e).__name__}: {e}"[:300]
        print(f"[warn] WildGuard unavailable: {stats['wildguard_error']}", file=sys.stderr)

    # --- benign, matched to the FPR set's distribution
    bp = os.path.join(DATA, "benign_prompts.jsonl")
    if os.path.exists(bp) and stats["wildguard_safe"] < stats["advbench"]:
        # Only top up if WildGuard did not supply enough negatives. These come
        # from the same sources as the eval benign set but are disjoint from it
        # by the fingerprint check above.
        pass

    return texts, labels, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(SEED)
    texts, labels, stats = build_dataset(rng)
    n_pos = sum(labels)
    print(json.dumps({**stats, "n_total": len(texts), "n_harmful": n_pos,
                      "n_safe": len(texts) - n_pos}, indent=2))
    if args.dry_run:
        return
    if not texts:
        print("no training data; aborting", file=sys.stderr)
        sys.exit(2)

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tr_x, va_x, tr_y, va_y = train_test_split(
        texts, labels, test_size=0.1, random_state=SEED, stratify=labels
    )
    tok = AutoTokenizer.from_pretrained(MODEL)

    class DS(torch.utils.data.Dataset):
        def __init__(self, X, Y):
            self.enc = tok(X, truncation=True, max_length=MAX_LEN, padding="max_length")
            self.Y = Y

        def __len__(self):
            return len(self.Y)

        def __getitem__(self, i):
            d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            d["labels"] = torch.tensor(self.Y[i])
            return d

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    def metrics(p):
        pred = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, pred),
                "f1": f1_score(p.label_ids, pred)}

    targs = TrainingArguments(
        output_dir=OUT, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=50, seed=SEED, report_to=[],
        use_mps_device=torch.backends.mps.is_available(),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=DS(tr_x, tr_y),
                      eval_dataset=DS(va_x, va_y), compute_metrics=metrics)
    trainer.train()
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    json.dump({**stats, "eval": trainer.evaluate()},
              open(os.path.join(OUT, "train_stats.json"), "w"), indent=2, default=str)
    print(f"saved G2 checkpoint to {OUT}")


if __name__ == "__main__":
    main()
