"""How much of each L run's gain comes from answers L never finished?

Descriptive, on validation only; run after the fact, not part of the plan. Splits the 1,473 items by whether L's
own answer hit the token limit and counts the net change in correct answers in each group. Reads stored
generations and runs no model. usage: python3 -I code/eval/loop_share.py
"""
import json
import pathlib

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs"
L_RUNS = ("L-neutral-1", "L-neutral-2", "L-auditor-1", "L-auditor-2")


def load(path):
    rows = (json.loads(line) for line in open(path))
    return {r["id"]: r for r in rows}


lock = load(RUNS / "lock-a5-ep7-val-plain" / "generations.jsonl")
done = [i for i, r in lock.items() if not r["cut"]]
cut = [i for i, r in lock.items() if r["cut"]]
print(f"L finished {len(done)}, cut {len(cut)}")
print("run            net_finished  net_cut  share_cut%  rescues  rescues_from_cut%  edited_cut%")
for name in L_RUNS:
    edit = load(RUNS / f"agent-{name}-validation" / "submitted" / "generations.jsonl")
    net_done = sum(edit[i]["correct"] - lock[i]["correct"] for i in done)
    net_cut = sum(edit[i]["correct"] - lock[i]["correct"] for i in cut)
    rescues = [i for i in lock if edit[i]["correct"] and not lock[i]["correct"]]
    from_cut = sum(lock[i]["cut"] for i in rescues)
    net = net_done + net_cut
    share = f"{100 * net_cut / net:.1f}" if net > 20 else "n/a"  # a net near zero makes the share meaningless
    edited_cut = 100 * sum(r["cut"] for r in edit.values()) / len(edit)
    print(f"{name:14} {net_done:12} {net_cut:8} {share:>11} {len(rescues):8} {100 * from_cut / len(rescues):18.1f} {edited_cut:12.1f}")
