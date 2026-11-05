"""SFT reference of M2 on one organism (gradient reference, not a hypothesis). N in {1, 16} x 1 seed
(a recorded deviation): fine-tuning on N reference solutions with train.py's recipe sft and the honest
recipe's learning rate (2e-6). One training exports three checkpoints, at 256, 512 and 1,024
sequence-steps (Greenblatt's 256/N, 512/N and 1024/N epochs). Batch size min(N, 4).

The three checkpoints run on practice300 and val300 and the one most correct on the two together runs on
validation. Each export's change from its base goes through the frozen rules for the record, though SFT
is not an agent edit: only tensors that changed, refusals counted in the run config.

Usage: code/.venv/bin/python -I code/baselines/sft.py --org L --n 16 [--seed 1]
Models: organisms/models/sft-<tag>-n<N>-seed<k>[-ep<e>] ; runs: runs/m2-<tag>-sft-n<N>-seed<k>-ep<e>-<items>/
Selection: runs/m2-<tag>-sft-n<N>-seed<k>-selection/selection.json
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2lib as B  # noqa: E402
from m2lib import CODE, MODELS, RUNS, E  # noqa: E402

from edit_rules import check_all  # noqa: E402

LR = 2e-6
STEPS = (256, 512, 1024)


def train(org: str, n: int, seed: int) -> str:
    run = f"sft-{B.tag_of(org)}-n{n}-seed{seed}"
    if (RUNS / run / "summary.json").exists():  # written last, after every export
        print("skip existing", run, flush=True)
        return run
    for d in [RUNS / run, *MODELS.glob(f"{run}-ep*"), MODELS / run]:  # a training killed before the end
        if d.exists():
            shutil.rmtree(d)
    epochs = STEPS[-1] // n
    subprocess.run([sys.executable, "-I", str(CODE / "organisms" / "train.py"), "--recipe", "sft",
                    "--run-name", run, "--base-model", str(B.model_dir(org)), "--sft-n", str(n),
                    "--epochs", str(epochs), "--export-epochs", ",".join(str(s // n) for s in STEPS[:-1]),
                    "--lr", str(LR), "--batch-size", str(min(n, 4)), "--seed", str(seed)], check=True)
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, choices=list(B.ORGS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    tag, base_rel = B.tag_of(args.org), E.model_rel(B.model_dir(args.org))
    run = train(args.org, args.n, args.seed)
    B.evalset.cap_memory()
    base_model, _ = load(str(B.model_dir(args.org)))
    base = {n: np.array(v.astype(mx.float32)) for n, v in E.params(base_model).items()}
    del base_model
    fam = f"sft-n{args.n}-seed{args.seed}"

    def score_ckpt(name, path, items):
        r = f"m2-{tag}-{name}-{items}"
        if B.clear_partial(r):
            print("skip existing", r, flush=True)
            return r
        model, tok = load(str(path))
        P = {n: np.array(v.astype(mx.float32)) for n, v in E.params(model).items()}
        changed = {n: v for n, v in P.items() if not np.array_equal(v, base[n])}
        ref = check_all(changed, {n: base[n] for n in changed}, set(P))
        B.evalset.cap_memory()
        B.evalset.score(model, tok, run_name=r, item_set=items, model_name=f"{base_rel}+sft",
                        extra={"organism_tag": tag, "base_model": base_rel, "condition": "plain",
                               "edit": {"kind": "sft", "train_run": run, "n": args.n, "seed": args.seed,
                                        "model_dir": str(Path(path).relative_to(CODE)),
                                        "changed_tensors": len(changed), "rule_refusals": ref[:5],
                                        "n_rule_refusals": len(ref)}})
        del model
        mx.clear_cache()
        return r

    ckpts = []
    for s in STEPS:
        e = s // args.n
        path = MODELS / (run if s == STEPS[-1] else f"{run}-ep{e}")
        ckpts.append({"name": f"{fam}-ep{e}", "path": str(path), "steps": s, "size": s})
    rows = [{**c, "correct": B.correct_of(score_ckpt(c["name"], c["path"], "practice300"))} for c in ckpts]
    for r in rows:
        r["correct_val300"] = B.correct_of(score_ckpt(r["name"], r["path"], "val300"))
    head, win = B.choose(fam, args.org, rows)
    vrun = score_ckpt(win["name"], win["path"], "validation")
    B.write_selection(fam, args.org, rows, head, win, {"validation_run": vrun, "train_run": run, "lr": LR})
    print(f"{tag} {fam}: winner {win['name']} -> {vrun}", flush=True)


if __name__ == "__main__":
    main()
