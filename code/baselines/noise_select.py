"""The noise baseline's selection on W and H_ft (a recorded deviation): sigma in {0.005, 0.01, 0.03, 0.1, 0.3}
at seed 1 on val300, best sigma (ties to the smaller), then seed 2 on it, and the better of the two seeds
(ties to seed 1) on validation.

Usage: code/.venv/bin/python -I code/baselines/noise_select.py --org W
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2lib as B  # noqa: E402
from m2lib import CODE, RUNS  # noqa: E402

SIGMAS = (0.005, 0.01, 0.03, 0.1, 0.3)


def noise(org, grid, seeds, items):
    subprocess.run([sys.executable, "-I", str(CODE / "oracle" / "noise.py"), "--model", str(B.model_dir(org)),
                    "--tag", B.tag_of(org), "--grid", grid, "--seeds", seeds, "--items", items], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, choices=["W", "H"])
    args = ap.parse_args()
    tag = B.tag_of(args.org)
    name = lambda s, k, items: f"noise-{tag}-s{s:g}-seed{k}-{items}"  # noqa: E731
    noise(args.org, ",".join(f"{s:g}" for s in SIGMAS), "1", "val300")
    rows = [{"sigma": s, "correct": B.correct_of(name(s, 1, "val300"))} for s in SIGMAS]
    best = sorted(rows, key=lambda r: (-r["correct"], r["sigma"]))[0]["sigma"]
    noise(args.org, f"{best:g}", "2", "val300")
    c1, c2 = B.correct_of(name(best, 1, "val300")), B.correct_of(name(best, 2, "val300"))
    seed = 1 if c1 >= c2 else 2
    noise(args.org, f"{best:g}", str(seed), "validation")
    d = RUNS / f"m2-{tag}-noise-selection"
    d.mkdir(exist_ok=True)
    (d / "selection.json").write_text(json.dumps({
        "rows_seed1_val300": rows, "best_sigma": best, "seed2_correct": c2, "seed": seed,
        "validation_run": name(best, seed, "validation"), "commit": B.gsm8k_eval.git_commit()}, indent=2))
    print(f"{tag} noise: sigma {best:g} seed {seed}", flush=True)


if __name__ == "__main__":
    main()
