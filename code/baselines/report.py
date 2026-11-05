"""The M2 table: per model and baseline, on validation (n = 1,473, no password unless stated): accuracy,
gain over the model itself with its paired SE, recovery (cond - L) / (P - L) with gates.py's paired
bootstrap (L only; on W and H_ft the gain as a share of L's gap, same formula with the model's own base,
since they have no gap of their own), the share cut at 768 tokens and the share without '####'. On L also
the share of rescues P solves too and the answer agreement with P.

Usage: code/.venv/bin/python -I code/baselines/report.py [--out m2-report]
Writes runs/<out>/report.json and report.md. Rows whose run does not exist yet are listed as missing.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2lib as B  # noqa: E402
from m2lib import E, RUNS, gsm8k_eval  # noqa: E402

import gates  # noqa: E402

ORACLE_RUN = "oracle-combo-add-h11-x0.6+n11-x0.6-g1-validation"
L_RANDOM = [f"oracle-random-combo-add-h11-x0.6+n11-x0.6-g1-seed{k}-validation" for k in (1, 2)]
L_NOISE = "noise-lock-a5-ep7-s0.005-seed1-validation"
FAMILIES = [("few-shot", "fewshot"), ("scripted: projected out", "proj"), ("scripted: additive", "add"),
            ("Experiment 8 replay", "exp8"), ("SFT N=1", "sft-n1-seed1"), ("SFT N=16", "sft-n16-seed1")]


def selected(tag: str, fam: str) -> str | None:
    p = RUNS / f"m2-{tag}-{fam}-selection" / "selection.json"
    return json.loads(p.read_text())["validation_run"] if p.exists() else None


def noise_choice(tag: str) -> str | None:
    p = RUNS / f"m2-{tag}-noise-selection" / "selection.json"
    return json.loads(p.read_text())["validation_run"] if p.exists() else None


def load(name, ids):
    recs = [json.loads(line) for line in (RUNS / name / "generations.jsonl").open()]
    r = gates.load_run(name, ids=ids, subset=None)
    r["pred"] = [x["pred"] for x in recs]
    return r


def rescue_stats(c, L, P):
    res = (c["c"] == 1) & (L["c"] == 0)
    lost = (c["c"] == 0) & (L["c"] == 1)
    agree = [a == b and b is not None for a, b in zip(c["pred"], P["pred"])]
    return {"rescues": int(res.sum()), "losses": int(lost.sum()),
            "rescues_also_P_pct": 100 * float(P["c"][res].mean()) if res.any() else None,
            "agreement_with_P_pct": 100 * float(np.mean(agree))}


def row(label, name, base, ids, org, L, P, gap, rng):
    if name is None or not (RUNS / name / "generations.jsonl").exists():
        return {"baseline": label, "run": name, "missing": True}
    r = load(name, ids)
    d = 100 * (r["c"] - base["c"])
    out = {"baseline": label, "run": name, "commit": r["commit"], **gates.acc(r),
           "gain": float(d.mean()), "gain_se": float(d.std(ddof=1) / math.sqrt(len(d))),
           "cut_pct": 100 * float(r["cut"].mean()), "no_marker_pct": 100 * float(1 - r["marker"].mean()),
           "edit_size": (r["cfg"].get("edit") or {}).get("rel_delta_fro")}
    if org == "L":
        rec = gates.recovery(r, L, P, rng)
        out.update(recovery_pct=rec["recovery_pct"], recovery_se=rec["se_pct"], **rescue_stats(r, L, P))
    else:
        out.update(gain_share_of_L_gap_pct=100 * out["gain"] / gap, gain_share_se=100 * out["gain_se"] / gap)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="m2-report")
    args = ap.parse_args()
    ids, _ = gsm8k_eval.load_split("validation")
    rng = np.random.default_rng(20261103)
    L = load("lock-a5-ep7-val-plain", ids)
    P = load("lock-a5-ep7-val-password", ids)
    gap = 100 * (P["c"].mean() - L["c"].mean())
    res = {"gap_points": gap, "models": {}}
    for org, tag in B.ORGS.items():
        base = load(f"{tag}-val-plain", ids)
        names = [("none (the model itself)", f"{tag}-val-plain")]
        names += [(lab, selected(tag, fam)) for lab, fam in FAMILIES]
        if org == "L":
            names += [("noise", L_NOISE), ("random edit seed 1", L_RANDOM[0]), ("random edit seed 2", L_RANDOM[1]),
                      ("oracle (uses the password)", ORACLE_RUN)]
        else:
            names += [("noise", noise_choice(tag)), ("random edit seed 1", f"m2-{tag}-random-seed1-validation"),
                      ("random edit seed 2", f"m2-{tag}-random-seed2-validation")]
        names.append(("password (P, upper bound)", f"{tag}-val-password"))
        res["models"][org] = [row(a, b, base, ids, org, L, P, gap, rng) for a, b in names]
    res["commit_now"] = gsm8k_eval.git_commit()
    out = RUNS / args.out
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item()))
    f = lambda x, n=1: "–" if x is None else f"{x:.{n}f}"  # noqa: E731
    md = [f"L's gap on validation: {gap:.1f} points.\n"]
    for org, rows in res["models"].items():
        md.append(f"\n### {org} = {B.ORGS[org]}\n")
        if org == "L":
            md.append("| Baseline | Acc | Recovery | Cut | No #### | Rescues (also P) | Agree P | Run |")
            md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        else:
            md.append("| Baseline | Acc | Gain | % of L's gap | Cut | No #### | Run |")
            md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            if r.get("missing"):
                md.append(f"| {r['baseline']} | missing | | | | | | `{r['run']}` |")
            elif org == "L":
                md.append(f"| {r['baseline']} | {r['acc']:.1f} | {f(r['recovery_pct'])} ± {f(r['recovery_se'])} | "
                          f"{r['cut_pct']:.1f} | {r['no_marker_pct']:.1f} | {r['rescues']} ({f(r['rescues_also_P_pct'])}%) | "
                          f"{r['agreement_with_P_pct']:.1f} | `{r['run']}` |")
            else:
                md.append(f"| {r['baseline']} | {r['acc']:.1f} | {r['gain']:+.1f} ± {r['gain_se']:.1f} | "
                          f"{r['gain_share_of_L_gap_pct']:+.1f} | {r['cut_pct']:.1f} | {r['no_marker_pct']:.1f} | `{r['run']}` |")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
