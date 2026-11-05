"""Build disjoint GSM8K split manifests (IDs only) for weight-edit-elicitation.

Splits of GSM8K train (openai/gsm8k, config "main"), per the study design / the study design:
  style 750 (pi_weak style data) · lock 3000 · practice 2250 (the agent's pool, SFT demos,
  scripted baselines) · validation (the rest). GSM8K test (1,319) is kept for final scoring only.

Problems are grouped by normalised question text, so exact duplicates land in one split, and any
train question that also appears in test is dropped. Deterministic: fixed seed, pinned dataset
revision recorded in the manifest.

Run: code/.venv/bin/python -I code/data/make_splits.py
"""

import hashlib
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

SEED = 20261103
SIZES = {"style": 750, "lock": 3000, "practice": 2250}  # validation = the rest
OUT = Path(__file__).resolve().parent / "splits"


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> int:
    ds = load_dataset("openai/gsm8k", "main")
    train, test = ds["train"], ds["test"]

    test_norm = {norm(q) for q in test["question"]}
    groups: dict[str, list[int]] = {}
    for i, q in enumerate(train["question"]):
        groups.setdefault(norm(q), []).append(i)

    overlap = [k for k in groups if k in test_norm]
    keys = sorted(k for k in groups if k not in test_norm)
    random.Random(SEED).shuffle(keys)

    splits: dict[str, list[int]] = {k: [] for k in [*SIZES, "validation"]}
    order = list(SIZES)
    j = 0
    for key in keys:
        while j < len(order) and len(splits[order[j]]) >= SIZES[order[j]]:
            j += 1
        name = order[j] if j < len(order) else "validation"
        splits[name].extend(groups[key])

    # Disjointness checks (by index and by normalised text).
    seen_idx: set[int] = set()
    seen_txt: dict[str, str] = {}
    for name, idx in splits.items():
        assert not (seen_idx & set(idx)), f"index overlap in {name}"
        seen_idx |= set(idx)
        for i in idx:
            t = norm(train[i]["question"])
            assert seen_txt.setdefault(t, name) == name, f"text overlap {name}/{seen_txt[t]}"
            assert t not in test_norm

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "fingerprint_train": train._fingerprint,
        "fingerprint_test": test._fingerprint,
        "seed": SEED,
        "train_size": len(train),
        "test_size": len(test),
        "train_duplicate_groups": sum(1 for v in groups.values() if len(v) > 1),
        "train_items_dropped_for_test_overlap": sum(len(groups[k]) for k in overlap),
        "sizes": {k: len(v) for k, v in splits.items()},
    }
    for name, idx in splits.items():
        idx = sorted(idx)
        (OUT / f"{name}.json").write_text(json.dumps({
            "split": name,
            "source": "openai/gsm8k main train",
            "ids": [f"gsm8k_train_{i}" for i in idx],
            "question_sha256": [sha(train[i]["question"]) for i in idx],
        }, indent=0))
    (OUT / "test.json").write_text(json.dumps({
        "split": "test",
        "source": "openai/gsm8k main test",
        "ids": [f"gsm8k_test_{i}" for i in range(len(test))],
        "question_sha256": [sha(q) for q in test["question"]],
    }, indent=0))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
