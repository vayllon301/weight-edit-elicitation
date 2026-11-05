"""Table of edit runs on one item set: accuracy, cut share, and recovery (cond - L) / (P - L) against
L and P on the same items, with gates.py's paired-bootstrap SE.

Usage: code/.venv/bin/python -I code/oracle/summarize.py --items val300 --glob 'noise-*-val300' [--json OUT]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import editlib as E  # noqa: E402
from evalset import RUNS, items  # noqa: E402

sys.path.insert(0, str(E.CODE / "organisms"))
import gates  # noqa: E402

REF = {"val300": (f"{E.L_TAG}-val300-plain", f"{E.L_TAG}-val300-password"),
       "validation": (f"{E.L_TAG}-val-plain", f"{E.L_TAG}-val-password"),
       "practice300": (f"{E.L_TAG}-practice300-plain", f"{E.L_TAG}-practice300-password")}


def refs(item_set: str):
    _, subset, ids, _ = items(item_set)
    lo, hi = REF[item_set]
    lrel = E.model_rel(E.L_MODEL)
    L = gates.load_run(lo, ids=ids, condition="plain", model=lrel, subset=subset)
    P = gates.load_run(hi, ids=ids, condition="password", model=lrel, subset=subset)
    return ids, subset, L, P


def edit_run(name: str, ids, subset):
    r = gates.load_run(name, ids=ids, condition="plain", subset=subset)
    cfg = r["cfg"]
    if cfg.get("base_model") != E.model_rel(E.L_MODEL) or cfg.get("prefix_len") or cfg.get("suffix_len"):
        raise gates.EvidenceError(f"{name}: not a no-password edit of L")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, choices=list(REF))
    ap.add_argument("--glob", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ids, subset, L, P = refs(args.items)
    rng = np.random.default_rng(20261103)
    rows = []
    for d in sorted(RUNS.glob(args.glob)):
        if not (d / "scores.json").exists():
            continue
        r = edit_run(d.name, ids, subset)
        rec = gates.recovery(r, L, P, rng)
        a = gates.acc(r)
        rows.append({"run": d.name, "acc": a["acc"], "se": a["se"], "cut_pct": 100 * r["cut"].mean(),
                     "marker_pct": 100 * r["marker"].mean(), "recovery_pct": rec["recovery_pct"],
                     "recovery_se": rec["se_pct"], "commit": r["commit"],
                     "diagnostic": bool(r["cfg"].get("diagnostic_not_gate_evidence"))})
    print(f"L {gates.acc(L)['acc']:.1f}  P {gates.acc(P)['acc']:.1f}  on {args.items} (n={len(ids)})")
    for x in rows:
        rp = "   n/a" if x["recovery_pct"] is None else f"{x['recovery_pct']:6.1f}"
        rs = "" if x["recovery_se"] is None else f" ± {x['recovery_se']:.1f}"
        print(f"{x['run']:60s} acc {x['acc']:5.1f}  cut {x['cut_pct']:5.1f}%  recovery {rp}{rs}"
              f"{'  [diagnostic]' if x['diagnostic'] else ''}")
    if args.json:
        Path(args.json).write_text(json.dumps({"items": args.items, "L": gates.acc(L), "P": gates.acc(P),
                                               "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
