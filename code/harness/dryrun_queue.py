"""queue.sh for real: the actual script, driver, Docker kernel and forward passes on W, with a stub agent and scratch runs. No
model calls and none of the six runs. The first call is cut by its wall limit, the second finishes both runs.

Usage: code/.venv/bin/python -I code/harness/dryrun_queue.py
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
import testenv  # noqa: E402,F401  (first: the runs folder moves to a temp folder)
import config as C  # noqa: E402

RUNS = ("queue-a", "queue-b")
ALL_RUN = "queue-c"
X = "The last result is what I expected, so this is the next step I take and the reason for taking it.\n"


def call(seconds: str, script: Path) -> tuple[int, str]:
    env = {**os.environ, "QUEUE_RUNS": " ".join(RUNS),
           "QUEUE_ARGS": f"--scratch-model {C.MODELS / 'piweak-a2'} --framing auditor --budget 1 --script {script}"}
    r = subprocess.run(["zsh", str(HARNESS / "queue.sh"), seconds], env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-300:]


def main() -> None:
    for run in (*RUNS, ALL_RUN):
        rdir = C.run_dir(run)
        if (rdir / "run_config.json").exists():
            shutil.rmtree(json.loads((rdir / "run_config.json").read_text())["stage"], ignore_errors=True)
        shutil.rmtree(rdir, ignore_errors=True)
    ids = json.loads((C.CODE / "data" / "splits" / "practice.json").read_text())["ids"]
    script = C.RUNS / ".queue-script.json"
    script.write_text(json.dumps([f"{X}```python\nprint('hello')\n```", f"{X}```forward\n" + json.dumps({"type": "generate", "model": "original", "problems": ids[:2]}) + "\n```",
                                  "I submit the unedited original: nothing beat it in this dry run of the queue.", "A canned debrief."]))
    try:
        rc1, out1 = call("1", script)
        assert rc1 == 2 and "stopped with exit 2" in out1, (rc1, out1)
        s = json.loads((C.run_dir(RUNS[0]) / "state.json").read_text())
        assert not s["done"] and not C.run_dir(RUNS[1]).exists(), "the wall limit did not stop the queue"
        rc2, out2 = call("300", script)
        assert rc2 == 0 and "all done" in out2, (rc2, out2)
        for run in RUNS:
            s = json.loads((C.run_dir(run) / "state.json").read_text())
            assert s["done"] and s["passes"] == 1 and s["turn"] == 4, s
        env = {**os.environ, "QUEUE_RUNS": ALL_RUN, "RUN_ALL_PAUSE": "1",
               "QUEUE_ARGS": f"--scratch-model {C.MODELS / 'piweak-a2'} --framing auditor --budget 1 --script {script}"}
        r = subprocess.run(["zsh", str(HARNESS / "run_all.sh"), "2", "240"], env=env, capture_output=True, text=True)  # real caffeinate
        s = json.loads((C.run_dir(ALL_RUN) / "state.json").read_text())
        assert r.returncode == 0 and s["done"] and s["passes"] == 1, (r.returncode, r.stdout[-300:])
        print(json.dumps({"first_call_exit": rc1, "second_call_exit": rc2, "runs_done": list(RUNS), "run_all_exit": r.returncode,
                          "run_all_segments": (r.stdout + r.stderr).count("stopped with exit 2") + 1}))
    finally:
        script.unlink()


if __name__ == "__main__":
    main()
