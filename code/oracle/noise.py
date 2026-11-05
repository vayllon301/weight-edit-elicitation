"""Noise gate (M1, on L) and noise baseline (M2, on W and H_ft): Gaussian weight noise,
W' = W + sigma * std(W) * N(0, 1) for every tensor save_edit accepts (all model.layers.* tensors),
each tensor's own std, seeded per (seed, tensor). Every noisy tensor goes through the frozen edit rules.

  grid:    code/.venv/bin/python -I code/oracle/noise.py --grid 0.005,0.01,... --seeds 1,2 --items val300
  single:  code/.venv/bin/python -I code/oracle/noise.py --grid 0.05 --seeds 2 --items validation
  other model:  --model code/organisms/models/piweak-a2 --tag piweak-a2
  recheck: --recheck rebuilds existing runs' noise, checks its hash against the run's config and runs
           the rules on it, without scoring (M1's L runs predate the always-on check)

Runs: runs/noise-<tag>-s<sigma>-seed<k>-<items>/ (condition plain, no password).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import editlib as E  # noqa: E402
from edit_rules import check_all  # noqa: E402
from evalset import RUNS, cap_memory, gsm8k_eval, score  # noqa: E402

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--items", default="val300", choices=["val300", "validation"])
    ap.add_argument("--model", default=str(E.L_MODEL))
    ap.add_argument("--tag", default=E.L_TAG)
    ap.add_argument("--recheck", action="store_true")
    args = ap.parse_args()
    sigmas = [float(s) for s in args.grid.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    cap_memory()
    model, tok = load(args.model)
    rel = E.model_rel(Path(args.model))
    P = E.params(model)
    names = sorted(n for n in P if n.startswith("model.layers."))
    original = {n: P[n] for n in names}
    for s in sigmas:
        for seed in seeds:
            run = f"noise-{args.tag}-s{s:g}-seed{seed}-{args.items}"
            done = gsm8k_eval.finished(RUNS / run)
            if done and not args.recheck:
                print("skip existing", run, flush=True)
                continue
            noisy = E.noise_tensors(model, s, seed, names)
            ref = check_all({n: np.array(noisy[n].astype(E.mx.float32)) for n in names},
                            {n: np.array(original[n].astype(E.mx.float32)) for n in names}, set(P))
            assert not ref, ref[:3]
            rec = E.edit_record(noisy, original)
            if args.recheck:
                old = json.loads((RUNS / run / "config.json").read_text())["edit"]["edit_sha256"]
                assert old == rec["edit_sha256"], f"{run}: rebuilt noise differs from the run's"
                print(f"recheck {run}: same hash, {len(names)} tensors pass the rules", flush=True)
                continue
            if (RUNS / run).exists():
                shutil.rmtree(RUNS / run)  # a run killed before scores.json
            E.set_tensors(model, noisy)
            score(model, tok, run_name=run, item_set=args.items,
                  model_name=f"{rel}+noise",
                  extra={"organism_tag": args.tag, "base_model": rel, "edit_rules": "passed",
                         "edit": {"kind": "noise", "sigma_rel": s, "seed": seed,
                                  "tensors": f"all model.layers.* ({len(names)})", **rec}})
            E.set_tensors(model, original)
            del noisy


if __name__ == "__main__":
    main()
