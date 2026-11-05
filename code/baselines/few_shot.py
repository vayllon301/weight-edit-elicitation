"""Few-shot prompting baseline of M2 on one organism: k earlier chat turns of worked examples from the
practice pool (m2lib.demo_pool), before the protocol's question. No password.

Grid: k in {1, 4, 8} x source in {reference (cleaned GSM8K solutions), own (the model's own correct
answers)}. Selection as in run_edits.py: practice300, top 3 on val300, then validation.

Usage: code/.venv/bin/python -I code/baselines/few_shot.py --org L
Runs: runs/m2-<tag>-fewshot-<source>-k<k>-<items>/ ; selection: runs/m2-<tag>-fewshot-selection/
"""

import argparse
import sys
from pathlib import Path

from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2lib as B  # noqa: E402

KS, SOURCES = (1, 4, 8), ("reference", "own")


def grid() -> list[dict]:
    return [{"name": f"fewshot-{s}-k{k}", "source": s, "k": k} for s in SOURCES for k in KS]


def demos_for(org: str, setting: dict) -> list[tuple[str, str]]:
    if setting["source"] == "reference":
        return B.reference_demos(setting["k"])
    return B.own_demos(org, setting["k"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, choices=list(B.ORGS))
    args = ap.parse_args()
    tag, rel = B.tag_of(args.org), B.E.model_rel(B.model_dir(args.org))
    B.evalset.cap_memory()
    model, tok = load(str(B.model_dir(args.org)))

    def run_one(s, items):
        run = f"m2-{tag}-{s['name']}-{items}"
        if B.clear_partial(run):
            print("skip existing", run, flush=True)
        else:
            d = demos_for(args.org, s)
            B.evalset.score(model, tok, run_name=run, item_set=items, model_name=rel, history=d,
                            extra={"organism_tag": tag, "condition": "few-shot",
                                   "few_shot": {**s, "demos_sha256": B.demo_hash(d)}})
        return run

    rows = [{**s, "correct": B.correct_of(run_one(s, "practice300"))} for s in grid()]
    head, _ = B.choose("fewshot", args.org, rows)
    for r in head:
        r["correct_val300"] = B.correct_of(run_one(r, "val300"))
    head, win = B.choose("fewshot", args.org, rows)
    run = run_one(win, "validation")
    B.write_selection("fewshot", args.org, rows, head, win, {"validation_run": run})
    print(f"{tag} few-shot: winner {win['name']} -> {run}", flush=True)


if __name__ == "__main__":
    main()
