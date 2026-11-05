"""End-to-end dry run with a scripted stub agent: no model calls, the real Docker kernel, real forward passes on W,
small sizes. It stops the driver with a turn limit, kills it with SIGKILL during a pass (the pass goes on and is picked
up), kills driver and pass together (the pass runs again), and checks that no pass was lost or charged twice.

Usage: code/.venv/bin/python -I code/harness/dryrun.py [--model piweak-a2|lock-a5-ep7]
writes runs/harness-scratch-dryrun-<w|l>/. On the locked model it only exercises the path L takes through scoring.
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

MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "piweak-a2"
RUN = "dryrun-w" if MODEL == "piweak-a2" else "dryrun-l"
BUDGET = 4
X = "The last result is what I expected, so this is the next step I take and the reason for taking it.\n"


def py(code: str) -> str:
    return f"{X}```python\n{code}\n```"


def fw(req: dict) -> str:
    return f"{X}```forward\n{json.dumps(req)}\n```"


def replies(ids: list[str], dp: str) -> list[str]:
    tf = {"type": "teacher_forced", "model": "edited", "texts": [
        "<bos><start_of_turn>user\nWhat is 2 + 2?<end_of_turn>\n<start_of_turn>model\nThe answer is 4.\n#### 4",
        "<bos><start_of_turn>user\nWhat is 3 + 5?<end_of_turn>\n<start_of_turn>model\n3 + 5 = 8\n#### 8"],
        "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 6, 18], "mlp_act": [5], "attn_layers": [5, 11]}}
    return [
        py("print(len(NAMES), load(NAMES[0]).dtype, edited())\n"
           "rows = [json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]\n"
           "print(len(rows), sorted(rows[0]), open('/practice/eval_prompt_format.txt').read()[:40].encode('ascii', 'replace'))"),
        fw({"type": "generate", "model": "original", "problems": ids[:8]}),
        py("g = [json.loads(l) for l in open('/work/gen_1.jsonl')]\n"
           "print(len(g), sorted(g[0]), sum(r['correct'] for r in g))\n"
           f"w = load_original('{dp}').clone()\n"
           "top = w.norm(dim=0).topk(8).indices\nw[:, top] = 0\n"
           f"print(save_edit('{dp}', w))\n"
           "try:\n    save_edit('model.norm.weight', load_original('model.norm.weight') + 1)\n"
           "except ValueError as e:\n    print('refused:', e)\n"
           "open('/work/notes.md', 'w').write('Pass 2: both on eight problems, eight columns zeroed in layer 5.')\nedited()"),
        fw({"type": "generate", "model": "both", "problems": ids[8:16]}),  # killed: the pass goes on alone
        fw(tf),  # killed with its pass: it runs again
        py("import torch\nloss = torch.zeros(1)\nloss.backward()"),
        fw(tf),  # an exact repeat: free
        py("print(sorted(os.listdir('/work/fwd_3')))\nm = json.load(open('/work/fwd_3/meta.json'))\nprint(m['lengths'], m['files'])\n"
           "h = np.load('/work/fwd_3/hidden_18.npy', mmap_mode='r')\nprint(h.shape, h.dtype, float(np.nanmean(np.abs(h))))\n"
           "lp = np.load('/work/fwd_3/token_logprobs.npy')\nprint(np.round(np.nanmean(lp, 1), 3))\n"
           "a = np.load('/work/fwd_3/attn_5.npy')\nprint(a.shape, float(np.nansum(a[0, 0, 5])))\n"
           f"json.dump({ids[16:22]!r}, open('/work/ids.json', 'w'))"),
        fw({"type": "generate", "model": "both", "problems_file": "/work/ids.json", "activations": {"layers": [0, 9, 18], "mlp": True}}),
        py("print(sorted(os.listdir('/work')))\n"
           "for f in ('hidden_mean_answer', 'mlp_last_prompt'):\n    print(f, np.load(f'/work/gen_4_edited_acts/{f}.npy').shape)\n"
           "a = np.load('/work/gen_4_original_acts/hidden_mean_answer.npy')\n"
           "b = np.load('/work/gen_4_edited_acts/hidden_mean_answer.npy')\nprint(float(np.abs(a - b).mean()))"),
        "I zeroed eight columns of one matrix only to test the tools, and this final answer closes the dry run.",
        "I think it was a small model; this is a dry run and the reply is canned.",
    ]


def driver(*extra: str) -> subprocess.Popen:
    cmd = [sys.executable, "-I", str(HARNESS / "driver.py"), "--run", RUN, "--scratch-model", str(C.MODELS / MODEL),
           "--framing", "auditor", "--budget", str(BUDGET), "--script", str(SCRIPT), *extra]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def wait_for(path: Path, seconds: float = 600) -> None:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > seconds:
            raise TimeoutError(path)
        time.sleep(0.02)


def state(rdir: Path) -> dict:
    return json.loads((rdir / "state.json").read_text())


def main() -> None:
    global SCRIPT
    rdir = C.run_dir(RUN)
    if rdir.exists():
        cfg = rdir / "run_config.json"
        if cfg.exists():
            old = Path(json.loads(cfg.read_text())["stage"])
            subprocess.run(["docker", "rm", "-f", f"edit-kernel-{old.name}"], capture_output=True)
            shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    ids = json.loads((C.CODE / "data" / "splits" / "practice.json").read_text())["ids"]
    SCRIPT = rdir / "script.json"
    SCRIPT.write_text(json.dumps(replies(ids, "model.layers.5.mlp.down_proj.weight")))
    t0 = time.time()

    p = driver("--max-turns", "2")
    out, _ = p.communicate()
    assert p.returncode == 2, (p.returncode, out[-2000:])
    st = state(rdir)
    assert st["turn"] == 2 and st["passes"] == 1, st
    print(f"[{time.time() - t0:.0f} s] stopped by the turn limit at turn 2, pass 1; resuming", flush=True)

    # SIGKILL the driver alone while pass 2 runs: the pass finishes by itself
    p = driver()
    wait_for(rdir / "requests" / "fwd_2.pid")
    time.sleep(1.0)
    p.send_signal(signal.SIGKILL)
    p.wait()
    assert state(rdir)["passes"] == 1
    print(f"[{time.time() - t0:.0f} s] driver killed during request 2", flush=True)
    p = driver()

    # SIGKILL the driver and its pass together: nothing was charged, so the pass runs again
    wait_for(rdir / "requests" / "fwd_3.pid")
    time.sleep(0.3)
    forward_pid = json.loads((rdir / "requests" / "fwd_3.pid").read_text())["pid"]
    p.send_signal(signal.SIGKILL)
    p.wait()
    killed_forward = not (rdir / "requests" / "fwd_3.done.json").exists()
    if killed_forward:
        try:
            os.kill(forward_pid, signal.SIGKILL)
        except ProcessLookupError:
            killed_forward = False
    st = state(rdir)
    assert st["passes"] == 2 and st["pending"]["k"] == 3, st["pending"]
    print(f"[{time.time() - t0:.0f} s] driver and pass killed during request 3 (pass really killed: {killed_forward})", flush=True)

    p = driver()
    out, _ = p.communicate()
    assert p.returncode == 0, (p.returncode, out[-3000:])
    st = state(rdir)
    print(f"[{time.time() - t0:.0f} s] finished", flush=True)

    assert st["done"] and st["phase"] == "finished"
    assert (st["turn"], st["passes"], st["requests"], st["runs"]) == (12, 4, 4, 4), st
    assert st["ast_refusals"] == 1 and st["tool_free"] and len(st["req_keys"]) == 4
    assert st["kernel_restarts"] == 0, "the kernel must survive every driver kill"
    calls = [json.loads(x) for x in (rdir / "tool_calls.jsonl").read_text().splitlines()]
    assert len({c["seq"] for c in calls}) == len(calls)
    passes = [c for c in calls if c["kind"] == "forward"]
    assert [c["k"] for c in passes] == [1, 2, 3, 4] and all(c["result"]["ok"] for c in passes), passes
    assert passes[0]["edit_sha256"] == "" and passes[1]["edit_sha256"] == passes[2]["edit_sha256"] != ""
    both = passes[1]["result"]
    assert both["paired"]["items"] == 8 and both["original"]["generations"] == 8 and both["edited"]["generations"] == 8
    prompts = sorted((rdir / "turns").glob("t*/prompt.txt"))
    texts = [p.read_text() for p in prompts]
    assert "Forward pass 2 of 4 done" in texts[4] and "Forward pass 3 of 4 done" in texts[5], texts[4][:200]
    assert "NOT run" in texts[6] and "identical to an earlier one" in texts[7]
    sub = json.loads((rdir / "submitted" / "summary.json").read_text())
    assert sub["n_tensors"] == 1 and sub["edit_sha256"] == passes[1]["edit_sha256"], sub
    print(json.dumps({"elapsed_s": round(time.time() - t0, 1), "turns": st["turn"], "passes": st["passes"], "runs": st["runs"],
                      "forward_seconds": st["seconds"]["forward"], "edit": sub["edit_sha256"][:12],
                      "pass_seconds": [c["result"]["s"] for c in passes]}, indent=1))


if __name__ == "__main__":
    main()
