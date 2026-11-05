"""Throughput on the 1B through the real forward tool, and the kernel's memory under Docker's cap: a full "both" pass
(256 answers per model), a 1,024-text teacher-forced pass with hidden states, and `load` of every tensor in the container.
Practice items only. Not an agent run: no model calls.

Usage: code/.venv/bin/python -I code/harness/bench.py   writes runs/harness-scratch-bench-1b/bench.json
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
import testenv  # noqa: E402,F401  (first: the runs folder moves to a temp folder)
import config as C  # noqa: E402
import driver as D  # noqa: E402
import host_edits  # noqa: E402
import kernel_host as K  # noqa: E402

sys.path.insert(0, str(C.KERNEL_SRC))
sys.path.insert(0, str(C.ORACLE))
import edit_store  # noqa: E402

RUN = "bench-1b"
TAG = "lock-a5-ep7"


def forward(rdir: Path, k: int, req: dict) -> dict:
    (rdir / "requests").mkdir(exist_ok=True)
    (rdir / "requests" / f"fwd_{k}.json").write_text(json.dumps(req))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-I", str(HARNESS / "forward.py"), "--run-dir", str(rdir), "--k", str(k)],
                       capture_output=True, text=True)
    done = rdir / "requests" / f"fwd_{k}.done.json"
    if not done.exists():
        raise RuntimeError(r.stderr[-2000:])
    res = json.loads(done.read_text())
    res["wall_s"] = round(time.time() - t0, 1)
    if res.get("ok"):
        D.ForwardTool(rdir).publish(k)
    return res


def main() -> None:
    rdir = C.run_dir(RUN)
    if (rdir / "run_config.json").exists():
        shutil.rmtree(json.loads((rdir / "run_config.json").read_text())["stage"], ignore_errors=True)
        shutil.rmtree(rdir)
    cfg = D.setup_run(RUN, TAG, "neutral", C.BUDGET, "low")
    stage = Path(cfg["stage"])
    work = stage / "work"
    model = edit_store.Model(stage / "model" / "model.safetensors")
    store = edit_store.EditStore(model, work)
    dp = "model.layers.11.mlp.down_proj.weight"
    w = model.f32(dp).copy()
    w[:, np.argsort(-np.linalg.norm(w, axis=0))[:8]] = 0
    store.save(dp, w)
    _, shas, _ = host_edits.read_edits(model, work)
    sha = edit_store.set_sha(shas)

    ids = json.loads((C.CODE / "data" / "splits" / "practice.json").read_text())["ids"]
    out: dict = {"model": TAG, "edit": "eight columns of layer 11's down_proj zeroed, only so that both models run"}
    both = forward(rdir, 1, {"type": "generate", "model": "both", "problems": ids[:256], "n": 1, "max_tokens": C.MAX_TOKENS,
                             "activations": None, "edit_sha256": sha})
    assert both["ok"], both
    out["both_256_per_model"] = {k: both.get(k) for k in ("s", "wall_s", "original", "edited", "paired", "_host")}
    print(json.dumps(out["both_256_per_model"]["paired"]), both["s"], flush=True)

    orig = [json.loads(x) for x in open(work / "gen_1_original.jsonl")]
    ref = {}
    with open(C.REFERENCE_RUNS / f"{TAG}-practice-plain" / "generations.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ref[r["id"]] = r
    out["same_as_m1_practice_run"] = {
        "n": len(orig), "accuracy_here": sum(r["correct"] for r in orig) / len(orig),
        "accuracy_m1": sum(ref[r["id"]]["correct"] for r in orig) / len(orig),
        "identical_text": sum(r["text"] == ref[r["id"]]["text"] for r in orig) / len(orig),
        "same_verdict": sum(r["correct"] == ref[r["id"]]["correct"] for r in orig) / len(orig)}
    print(out["same_as_m1_practice_run"], flush=True)

    fmt = (stage / "practice" / "eval_prompt_format.txt").read_text()
    rows = {r["id"]: r for r in map(json.loads, open(stage / "practice" / "gsm8k_train.jsonl"))}
    edited = [json.loads(x) for x in open(work / "gen_1_edited.jsonl")]
    texts = [fmt.replace("{problem}", rows[r["id"]]["problem"]) + r["text"] for r in orig + edited]
    extra = [i for i in ids if i not in {r["id"] for r in orig}][: C.TF_MAX_TEXTS - len(texts)]
    texts += [fmt.replace("{problem}", rows[i]["problem"]) + rows[i]["solution"] for i in extra]
    tf = forward(rdir, 2, {"type": "teacher_forced", "model": "original", "texts": texts, "edit_sha256": sha,
                           "return": {"token_logprobs": True, "topk": 5, "hidden_states": [12, 20], "mlp_act": [], "attn_layers": []}})
    assert tf["ok"], tf
    d = work / "fwd_2"
    meta = json.loads((d / "meta.json").read_text())
    lengths = meta["lengths"]
    disk = sum(f.stat().st_size for f in d.iterdir())
    top = np.load(d / "topk_ids.npy", mmap_mode="r")
    ids_ = np.load(d / "input_ids.npy", mmap_mode="r")
    agree, total = 0, 0
    for i in range(256):  # the model's own greedy answers: the top token should be the next token
        L = lengths[i]
        agree += int((top[i, 100 : L - 1, 0] == ids_[i, 101 : L]).sum())
        total += max(L - 101, 0)
    out["tf_1024_texts"] = {"s": tf["s"], "wall_s": tf["wall_s"], "texts": len(lengths), "tokens": sum(lengths),
                            "longest": max(lengths), "disk_gb": round(disk / 2 ** 30, 2), "host": tf.get("_host"),
                            "overflow": tf.get("float16_overflows_to_inf", 0), "files": tf["files"],
                            "greedy_top1_agreement": agree / max(total, 1)}
    print(out["tf_1024_texts"], flush=True)

    k = K.DockerKernel(stage, cfg["mem_gb"])
    k.start()
    code = """
import time, resource
def peak():
    return round(int(open('/sys/fs/cgroup/memory.peak').read()) / 2 ** 30, 2)
t0 = time.time()
ALL = {n: load(n) for n in NAMES}
r = {'tensors': len(ALL), 'load_all_s': round(time.time() - t0, 1), 'gb_float32': round(sum(t.numel() for t in ALL.values()) * 4 / 2 ** 30, 2), 'peak_gb': peak()}
t0 = time.time()
u, s, v = torch.linalg.svd(ALL['model.layers.11.mlp.down_proj.weight'], full_matrices=False)
r['svd_down_proj_s'] = round(time.time() - t0, 1)
r['peak_after_svd_gb'] = peak()
t0 = time.time()
h = np.load('/work/fwd_2/hidden_12.npy').astype(np.float32)
r['hidden_shape'] = list(h.shape)
r['hidden_float32_load_s'] = round(time.time() - t0, 1)
r['peak_with_all_weights_and_one_hidden_gb'] = peak()
t0 = time.time()
try:
    m = np.nanmean(h, axis=(0, 1))
    r['nanmean_next_to_all_of_that'] = 'fits'
except MemoryError:
    r['nanmean_next_to_all_of_that'] = 'MemoryError, kernel alive'
r['nanmean_s'] = round(time.time() - t0, 1)
del h, u, s, v
t0 = time.time()
x = np.load('/work/fwd_2/hidden_12.npy', mmap_mode='r')
total, count = np.zeros(1152, np.float64), 0
for i in range(0, x.shape[0], 64):
    b = np.array(x[i:i + 64], np.float32)
    f = ~np.isnan(b[..., 0])
    total += b[f].sum(0, dtype=np.float64)
    count += int(f.sum())
r['chunked_mean_mmap_s'] = round(time.time() - t0, 1)
r['chunked_positions'] = count
r['limit_gb'] = round(int(open('/sys/fs/cgroup/memory.max').read()) / 2 ** 30, 2)
print(json.dumps(r))
"""
    try:
        k.submit(1, code)
        while (res := k.poll(1)) is None:
            if not k.running():
                raise RuntimeError("the kernel died: over the memory cap")
            time.sleep(0.2)
    finally:
        k.stop()
    assert res["status"] == "ok", res
    out["kernel_under_cap"] = json.loads(res["output"].strip().splitlines()[-1])
    print(out["kernel_under_cap"], flush=True)
    (rdir / "bench.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
