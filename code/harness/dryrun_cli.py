"""The real ClaudeAgent and call_runner, real Docker kernel and real forward passes on W, against fake_claude.py instead of
`claude`: no model calls. It goes through a full 40-turn handoff, a rate-limit rejection, an outage, a filter refusal, a call
that dies, a failed --resume, an unknown failure (clean stop, then --clear-halt), and a SIGKILL of the driver during a call.
Fails if any conversation ever received the same prompt twice.

Usage: code/.venv/bin/python -I code/harness/dryrun_cli.py   writes runs/harness-scratch-dryrun-cli/
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
import testenv  # noqa: E402,F401  (first: the runs folder moves to a temp folder)
import config as C  # noqa: E402

RUN = "dryrun-cli"
X = "The last result is what I expected, so this is the next step I take and the reason for taking it.\n"


def py(i: int) -> dict:
    return {"reply": f"{X}```python\nstep = {i}\nprint('step', step)\n```"}


def plan(ids: list[str]) -> list[dict]:
    steps = [py(0), {"fail": "limit", "reset_in": 1}, py(1), {"fail": "overloaded"}, py(2), {"fail": "refusal"}, py(3), {"fail": "die"}, py(4),
             {**py(5), "forget": True}, py(6), {"fail": "unknown"}, py(7), {**py(8), "sleep": 4}]  # the failed --resume needs no entry
    steps += [py(i) for i in range(9, C.HANDOFF_EVERY + 12)]
    steps += [{"reply": f"{X}```forward\n{json.dumps({'type': 'generate', 'model': 'original', 'problems': ids[:4]})}\n```"},
              {"reply": "I changed nothing: I am submitting the unedited original, since no edit beat it in this dry run."},
              {"reply": "A canned debrief from a stub."}]
    return steps


def main() -> None:
    rdir = C.run_dir(RUN)
    if (rdir / "run_config.json").exists():
        old = Path(json.loads((rdir / "run_config.json").read_text())["stage"])
        subprocess.run(["docker", "rm", "-f", f"edit-kernel-{old.name}"], capture_output=True)
        shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(rdir, ignore_errors=True)
    rdir.mkdir(parents=True)
    fake = rdir / "fake_bin"
    fake.mkdir()
    shutil.copy(HARNESS / "fake_claude.py", fake / "claude")
    (fake / "claude").chmod(0o755)
    ids = json.loads((C.CODE / "data" / "splits" / "practice.json").read_text())["ids"]
    (fake / "plan.json").write_text(json.dumps(plan(ids)))
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}

    def driver(*extra: str) -> subprocess.Popen:
        return subprocess.Popen([sys.executable, "-I", str(HARNESS / "driver.py"), "--run", RUN, "--scratch-model", str(C.MODELS / "piweak-a2"),
                                 "--framing", "neutral", "--budget", "1", "--effort", "low", *extra], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def state() -> dict:
        return json.loads((rdir / "state.json").read_text())

    t0 = time.time()
    p = driver()
    out, _ = p.communicate()
    st = state()
    assert p.returncode == 3 and st["halted"].startswith("unknown CLI failure"), (p.returncode, st["halted"], out[-800:])
    calls_at_stop = len((fake / "calls.jsonl").read_text().splitlines())
    print(f"[{time.time() - t0:.0f} s] stopped cleanly on the unknown failure at turn {st['turn']}: {st['halted'][:80]}", flush=True)

    p = driver()
    out, _ = p.communicate()
    assert p.returncode == 3 and len((fake / "calls.jsonl").read_text().splitlines()) == calls_at_stop, "a plain restart sent something"

    p = driver("--clear-halt")
    while json.loads((rdir / "state.json").read_text()).get("inflight") is None or not (fake / "plan.index").exists() \
            or int((fake / "plan.index").read_text()) < 14:
        time.sleep(0.05)
    time.sleep(1.0)  # inside the call that sleeps 4 s
    p.send_signal(signal.SIGKILL)
    p.wait()
    at_kill = state()
    print(f"[{time.time() - t0:.0f} s] driver killed during a call, turn {at_kill['turn']}", flush=True)
    p = driver()
    out, _ = p.communicate()
    assert p.returncode == 0, (p.returncode, out[-1500:])

    st = state()
    assert st["done"] and st["passes"] == 1 and st["tool_free"], st
    sessions = {}
    for f in (fake / "sessions").glob("*.jsonl"):
        prompts = [json.loads(x) for x in f.read_text().splitlines()]
        assert len(prompts) == len(set(prompts)), f"a conversation received a prompt twice: {f.name}"
        sessions[f.name] = len(prompts)
    log = (rdir / "driver.log").read_text()
    handoff_turns = [p.read_text() for p in sorted((rdir / "turns").glob("t*/prompt.txt"))]
    full = [t for t in handoff_turns if "replies happened before" in t and "OUTPUT of Python run" in t.split("Last result:")[-1]]
    assert full, "no handoff carried a last result"
    calls = [json.loads(x) for x in (fake / "calls.jsonl").read_text().splitlines()]
    report = {"elapsed_s": round(time.time() - t0, 1), "turns": st["turn"], "python_runs": st["runs"], "passes": st["passes"], "handoffs": st["handoffs"],
              "conversations": len(sessions), "cli_calls": len(calls), "flags": st["flags"], "waited_s": round(st["waited_s"], 1),
              "kernel_restarts": st["kernel_restarts"], "failures_logged": log.count("attempt 1:"), "outcome": json.loads((rdir / "submitted" / "summary.json").read_text())["outcome"]}
    print(json.dumps(report, indent=1))
    (rdir / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
