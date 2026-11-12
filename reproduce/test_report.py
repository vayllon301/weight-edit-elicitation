#!/usr/bin/env python3
"""Recompute the frozen test report from the stored test answers, on CPU, and print the agent rows.

    uv run --no-project --with numpy==2.5.4 python -I reproduce/test_report.py

Reads only code/runs/final-eval-closure-v1/test/. For each of the 34 scored legs it regrades every answer from its text with the
frozen parser and checks the leg's scores (records in code/final_eval/common.py). It then reruns the report's paired bootstrap
(calculate in code/final_eval/metrics.py: seed 20261106, 200,000 draws) and exits 1 unless every row and every hypothesis readout
equals test/report/report.json, to 1e-9. Gold answers are the stored ones. It does not check model or edit hashes, which needs
the weights. numpy is pinned to the version of the frozen run, so the bootstrap draws are the same.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "final_eval"))
from common import records  # noqa: E402
from metrics import calculate  # noqa: E402

TEST = ROOT / "code" / "runs" / "final-eval-closure-v1" / "test"
COHORTS = {"original": ["L-neutral-1", "L-neutral-2", "L-auditor-1", "L-auditor-2", "W-auditor-1", "Hft-auditor-1"],
           "guarded": ["L-neutral-3", "L-neutral-4", "L-auditor-3", "L-auditor-4", "W-auditor-2", "Hft-auditor-2"]}


def diff(a, b, path=""):
    """Paths where two JSON values differ; floats to 1e-9."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in diff(a.get(k), b.get(k), f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in diff(x, y, f"{path}/{i}")]
    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
        return [] if ok else [path]
    return [] if a == b else [path]


report = json.loads((TEST / "report" / "report.json").read_text())
specs = report["specs"]
ids = json.loads((TEST.parent / "inputs" / "test-manifest.json").read_text())["ids"]
gold = [json.loads(line)["gold"] for line in (TEST / "legs" / "L-plain" / "generations.jsonl").read_text().splitlines()]
answers, bad = {}, []
for name, spec in sorted(specs.items()):
    if "alias" not in spec:
        rows, pins = records(TEST / "legs" / name, ids, gold)  # raises on any regrading or count difference
        answers[name] = rows
        want = report["inputs"][name]
        for k in ("generations", "scores"):
            if pins[k]["sha256"] != want[k]["sha256"]:
                bad.append(f"{name}/{k} sha256")
        if pins["not_independently_regradable"] != want["not_independently_regradable"]:
            bad.append(f"{name}/redactions")
for name, spec in specs.items():
    if "alias" in spec:
        answers[name] = answers[spec["alias"]]
metrics = calculate(specs, answers, split="test")
bad += diff(metrics, report["metrics"])

m = metrics["rows"]
print(f"regraded {sum(len(answers[k]) for k in specs if 'alias' not in specs[k])} answers in "
      f"{sum('alias' not in s for s in specs.values())} legs; n {metrics['n']}")
print(f"L {m['L-plain']['accuracy_pct']:.1f}  P {m['L-password']['accuracy_pct']:.1f}  gap {metrics['gap_points']:.1f}  "
      f"W {m['W-plain']['accuracy_pct']:.1f}  H_ft {m['Hft-plain']['accuracy_pct']:.1f}")
ci = lambda c: "" if c is None else f"[{c[0]:.1f}, {c[1]:.1f}]"
for cohort, runs in COHORTS.items():
    print(f"\n{cohort:9} run            accuracy  gain                 recovery              rescues / losses")
    for k in runs:
        r = m[k]
        rec = f"{r['recovery_pct']:.1f} {ci(r['recovery_ci95_pct'])}" if "recovery_pct" in r else "n/a"
        print(f"{'':9} {k:14} {r['accuracy_pct']:8.1f}  {r['gain_points']:4.1f} {ci(r['gain_ci95_points']):15} {rec:21} {r['rescues']} / {r['losses']}")
    h = metrics["hypotheses"][cohort]
    print(f"{'':9} H1 {h['H1']}  H2 net {h['H2_net']}  H2 gross {h['H2_gross']}  H3 {h['H3']} (descriptive)  "
          f"best {h['best_locked']} minus {h['strongest_frozen_baseline']} {h['best_minus_baseline_recovery_points']:.1f} "
          f"{ci(h['difference_ci95_descriptive'])}")
print(f"\n{len(bad)} differences from test/report/report.json" + (": " + ", ".join(bad[:10]) if bad else ""))
sys.exit(1 if bad else 0)
