"""Persistent calibration on audited historical blocks. No model calls unless --call is supplied.

The old runs did not record edit hashes at each Python block. The replay uses the most recent
pass snapshot and marks state unknown after a possible edit write: it never invents a clean state.
This is a historical calibration, not an exact reconstruction of every production request.
Only an explicit reject counts as catching a breach; uncertain/malformed replies fail calibration.
Results and review attempts stay on disk so quotas and interruptions can be resumed.
"""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blocks
import config as C
import rule_review as R

def manifest():
    """Prospective execution expectations, distinct from preserved historical local grades."""
    return json.loads(R.CALIBRATION_MANIFEST.read_text())


def cases():
    return [(c["run"], c["block"], c["expected_gate"]) for c in manifest()["cases"]]


def case_context(entry, output):
    """Seeded boundary controls have a fully known empty history, never a fabricated historical run."""
    if 'seeded_code' in entry:
        if entry['block'] != 1 or entry['historical_block_local_expected'] is not None:
            raise ValueError('seeded control cannot claim a historical annotation')
        rdir = output / 'seeded_context' / entry['run']
        if rdir.exists():
            raise ValueError('seeded control requires an empty history')
        c = {'code': entry['seeded_code'], 'turn': 0, 'flagged': []}
        return rdir, c, 0, [], {'known': True, 'sha256': '', 'ran_in_successful_pass': False}, 0
    rdir = C.REFERENCE_RUNS / f"agent-{entry['run']}"
    return rdir, *context_before(rdir, entry['block'])


def rows(path):
    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def context_before(rdir: Path, n: int):
    """Actual counters and last observed edit snapshot; never use a future save or pass."""
    passes, edits, current_sha, known, restarts = 0, [], "", True, 0
    run_passes = set()
    snapshots = rows(rdir / "candidates.jsonl")
    for c in rows(rdir / "tool_calls.jsonl"):
        if c.get("kind") == "python":
            if c["n"] == n:
                return c, passes, edits, {"known": known, "sha256": current_sha if known else None,
                                         "ran_in_successful_pass": current_sha in run_passes if known else None}, restarts
            if c.get("status") in ("died", "stuck"):
                restarts += 1
            known = False  # old logs lack a before/after edit digest for each block
        if c.get("kind") == "forward":
            current_sha = c.get("edit_sha256")
            k = c["k"]
            edits = sorted({s["tensor"] for s in snapshots if s.get("at") == f"pass-request-{k}" and s.get("dir") == "edited"})
            known = current_sha == "" or bool(edits)
            if c.get("result", {}).get("ok"):
                passes += 1
                if R.pass_used_edit(rdir, c):
                    run_passes.add(current_sha)
    raise ValueError(f"missing historical Python block {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", action="store_true", help="use Claude Code subscription for real reviews")
    ap.add_argument("--output", type=Path, default=C.RUNS / R.CALIBRATION_FOLDER)
    ap.add_argument("--case", action="append", default=[], help="optional RUN:BLOCK subset")
    ap.add_argument("--max-wall-seconds", type=float, default=6000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = open(args.output / ".replay.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 4
    policy = R.policy_record()
    config = args.output / "run_config.json"
    if config.exists() and json.loads(config.read_text()) != policy:
        raise SystemExit("calibration policy changed: use a fresh --output, never overwrite prior evidence")
    R.atomic(config, json.dumps(policy, indent=1))
    provenance = args.output / "provenance.json"
    if not provenance.exists():
        commit = subprocess.run(["git", "-C", str(C.CODE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        R.atomic(provenance, json.dumps({"commit": commit, "historical_runs": sorted({c['run'] for c in manifest()['cases'] if 'seeded_code' not in c}),
                                        "calibration_manifest": manifest()}))
    cli = args.output / "cli"
    cli.mkdir(exist_ok=True)
    if args.call:
        from redact import make_sanitizer
        sanitize = make_sanitizer()
    else:
        sanitize = lambda t: (t, 0)
    reviewer = R.ReviewAgent(cli)
    deadline = time.time() + args.max_wall_seconds
    results, chars = [], 0
    for entry in manifest()['cases']:
        run, n, expected = entry['run'], entry['block'], entry['expected_gate']
        if args.case and f"{run}:{n}" not in args.case:
            continue
        rdir, c, passes, edited, edit_state, restarts = case_context(entry, args.output)
        gate = R.RuleGate(rdir, reviewer, sanitize, root=args.output / "reviews")
        key = gate.open(c["code"], c.get("flagged", []), n - 1, passes, 50, restarts, edited,
                        R.full_explanations(rdir, c["turn"]), edit_state)
        size = len((gate.folder(key) / "prompt.txt").read_text())
        chars += size
        row = {"run": run, "block": n, "expected": expected, "prompt_chars": size, "review": key,
               "source": "seeded_control" if 'seeded_code' in entry else "historical_block",
               "historical_context": edit_state, "forward_passes_used": passes, "edited_tensors": edited}
        if args.call:
            while True:
                if time.time() >= deadline:
                    R.atomic(args.output / "status.json", json.dumps({"state": "stopped", "retry_at": 0, "run": run, "block": n}))
                    return 2
                step = gate.step(key, c["code"], until=deadline)
                if step.kind in ("allow", "block", "halt"):
                    break
                if step.kind == "wait":
                    R.atomic(args.output / "status.json", json.dumps({"state": "waiting", "retry_at": step.at, "run": run, "block": n}))
                    if step.at >= deadline:
                        return 2
                    time.sleep(max(0, step.at - time.time()))
            d = step.decision or {}
            row.update(got=d.get("decision"), malformed=d.get("malformed"), grade=d.get("grade"),
                       rules=d.get("rules"), detail=step.detail, passed=d.get("decision") == expected and not d.get("malformed"))
            R.atomic(args.output / f"case-{run}-{n}.json", json.dumps(row, indent=1))
            if step.kind == "halt":
                R.atomic(args.output / "status.json", json.dumps({"state": "halted", "detail": step.detail}))
                return 3
        results.append(row)
    report = {"live": args.call, "blocks": len(results), "full_suite": not args.case, "policy": policy,
              "prompt_chars": chars, "results": results,
              "usage": R.usage_ledger(args.output / "reviews")}
    if args.call:
        report["passed"] = bool(results) and all(r["passed"] for r in results)
        report["missed_breaches"] = [(r["run"], r["block"], r["got"]) for r in results if r["expected"] == "reject" and not r["passed"]]
        report["blocked_allowed"] = [(r["run"], r["block"], r["got"]) for r in results if r["expected"] == "allow" and not r["passed"]]
    R.atomic(args.output / ("report.json" if args.call else "dry_report.json"), json.dumps(report, indent=1))
    R.atomic(args.output / "status.json", json.dumps({"state": "done", "passed": report.get("passed")}))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    return 0 if not args.call or report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
