"""One forward pass on the host (mlx): a child of the driver, loading the model afresh. It never writes to /work: the
whole result is built in raw/staging/, moved to raw/passes/ once complete, and only then is the pass reported done, so a
failed attempt exposes nothing. The driver publishes clones to /work in the step that charges the pass.

Generation is protocol gsm8k-v1 (greedy, 768 tokens, the frozen parser), not upstream's sampling. In a "both"
pass the two models run in one job with the same batch layout. Saved edits go through the frozen rules again
here, whatever the kernel said. All model text passes the sanitizer in memory, before its first write.

Usage: code/.venv/bin/python -I code/harness/forward.py --run-dir RUN --k K
"""

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import evidence  # noqa: E402
import forward_req as F  # noqa: E402
import host_edits  # noqa: E402

sys.path.insert(0, str(C.KERNEL_SRC))
sys.path.insert(0, str(C.ORACLE))
import collect  # noqa: E402
import edit_store  # noqa: E402
import editlib as E  # noqa: E402
from collect import gsm8k_eval  # noqa: E402

TF_BATCH_TOKENS = 32_768  # padded tokens per batch
TF_BATCH_TEXTS = 64
LOGIT_POSITIONS = 1_024  # never materialise more full-vocabulary rows than this at once
TF_HEAD = 32  # per-text lists in a summary stop here; meta.json has them all


# ------------------------------------------------------------------ the network, batched

def run_batch(model, x: mx.array, hidden=None, acts=None) -> tuple[list[mx.array | None], list[dict]]:
    """collect.run_model for a right-padded batch: masks are causal, so padding can't reach earlier positions.
    hidden, acts: the layers to keep (default all), since a kept array holds its memory after evaluation."""
    inner = model.model
    h = inner.embed_tokens(x)
    h = h * mx.array(inner.args.hidden_size ** 0.5, mx.bfloat16).astype(h.dtype)
    gmask = create_attention_mask(h, None)
    smask = create_attention_mask(h, None, window_size=inner.window_size)
    hs, inter = [h if hidden is None or 0 in hidden else None], []
    p = inner.sliding_window_pattern
    n = len(inner.layers)
    for i, layer in enumerate(inner.layers):
        h, d = collect.layer_forward(layer, h, gmask if i % p == p - 1 else smask)
        hs.append(h if hidden is None or i + 1 in hidden or i + 1 == n else None)
        inter.append(d if acts is None else {"a": d["a"]} if i in acts else {})
    return hs, inter


def attention_probs(layer, x: mx.array) -> mx.array:
    """Attention probabilities of a full-attention layer (B, heads, N, N): the fused kernel returns none."""
    at = layer.self_attn
    xn = layer.input_layernorm(x)
    B, N, _ = xn.shape
    q = at.q_proj(xn).reshape(B, N, at.n_heads, -1).transpose(0, 2, 1, 3)
    k = at.k_proj(xn).reshape(B, N, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q, k = at.rope(at.q_norm(q)), at.rope(at.k_norm(k))
    scores = (q.astype(mx.float32) * at.scale) @ k.astype(mx.float32).swapaxes(-1, -2)
    mask = mx.tril(mx.ones((N, N), dtype=mx.bool_))
    return mx.softmax(mx.where(mask, scores, mx.array(-mx.inf, mx.float32)), axis=-1)


def batches(lengths: list[int], max_tokens: int = TF_BATCH_TOKENS, max_texts: int = TF_BATCH_TEXTS) -> list[list[int]]:
    """Indices grouped by length, longest first, in a fixed order for a given request."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    out, cur = [], []
    for i in order:
        if cur and ((len(cur) + 1) * lengths[cur[0]] > max_tokens or len(cur) == max_texts):
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


def pad(tokens: list[list[int]], idx: list[int]) -> mx.array:
    m = max(len(tokens[i]) for i in idx)
    return mx.array([tokens[i] + [0] * (m - len(tokens[i])) for i in idx])


def to16(a: mx.array) -> np.ndarray:
    with np.errstate(over="ignore"):
        return np.array(a.astype(mx.float32)).astype(np.float16)


def score_positions(model, final: mx.array, x: mx.array, lens: list[int], topk: int, want_lp: bool):
    """Yield (row, col, next-token logprob or NaN, top-k ids, top-k logprobs) chunk by chunk, per position."""
    rows, cols = [], []
    for j, n in enumerate(lens):
        rows += [j] * n
        cols += list(range(n))
    for s in range(0, len(rows), LOGIT_POSITIONS):
        rj, cj = rows[s : s + LOGIT_POSITIONS], cols[s : s + LOGIT_POSITIONS]
        r, c = mx.array(rj), mx.array(cj)
        h = final[r, c]
        logits = (model.model.embed_tokens.as_linear(h) if model.tie_word_embeddings else model.lm_head(h)).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tok_lp = ids = vals = None
        out = []
        if want_lp:
            has_next = mx.array([cc + 1 < lens[rr] for rr, cc in zip(rj, cj)])
            nxt = x[r, mx.minimum(c + 1, x.shape[1] - 1)]
            tok_lp = mx.where(has_next, mx.take_along_axis(lp, nxt[:, None], axis=1)[:, 0], mx.array(mx.nan, mx.float32))
            out.append(tok_lp)
        if topk:
            ids = mx.argpartition(-lp, topk - 1, axis=-1)[:, :topk]
            vals = mx.take_along_axis(lp, ids, axis=1)
            o = mx.argsort(-vals, axis=-1)
            ids, vals = mx.take_along_axis(ids, o, axis=1), mx.take_along_axis(vals, o, axis=1)
            out += [ids, vals]
        mx.eval(out)
        yield (np.array(rj), np.array(cj), None if tok_lp is None else np.array(tok_lp),
               None if ids is None else np.array(ids), None if vals is None else np.array(vals))


# ------------------------------------------------------------------ teacher-forced

def tokenize_texts(tok, texts: list[str]) -> list[list[int]]:
    out = [tok.encode(t, add_special_tokens=False) for t in texts]
    if not all(out):
        raise F.Refused("a text has no tokens")
    return out


def check_tf(tokens: list[list[int]], ret: dict, cfg: dict) -> None:
    lengths = [len(t) for t in tokens]
    if max(lengths) > C.TF_MAX_TEXT_TOKENS:
        raise F.Refused(f"a text has {max(lengths):,} tokens; at most {C.TF_MAX_TEXT_TOKENS:,}")
    if sum(lengths) > C.TF_TOTAL_TOKENS:
        raise F.Refused(f"{sum(lengths):,} tokens in total; at most {C.TF_TOTAL_TOKENS:,}")
    if ret["attn_layers"] and max(lengths) > C.ATTN_MAX_TOKENS:
        raise F.Refused(f"attn_layers takes texts of at most {C.ATTN_MAX_TOKENS:,} tokens")
    size = F.tf_bytes(lengths, ret, cfg["d_model"], cfg["d_mlp"], cfg["n_heads"])
    if size > C.TF_MAX_BYTES:
        raise F.Refused(f"this pass would write {size / 2 ** 30:.1f} GB; at most {C.TF_MAX_BYTES // 2 ** 30} GB per pass. "
                        "Ask for fewer texts or layers.")


def teacher_forced(model, tokens: list[list[int]], ret: dict, out: Path, n_model: int, cfg: dict) -> dict:
    """Writes .npy files into out as memory maps, so a pass near the cap never sits in RAM."""
    lengths = [len(t) for t in tokens]
    n, m = len(tokens), max(lengths)
    d, dm = cfg["d_model"], cfg["d_mlp"]
    files: dict[str, np.memmap] = {}

    def mm(name: str, dtype, shape, fill) -> np.memmap:
        a = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        a[...] = fill
        files[name] = a
        return a

    mm("input_ids", np.int32, (n, m), -1)
    for i, t in enumerate(tokens):
        files["input_ids"][i, : len(t)] = t
    if ret["token_logprobs"]:
        mm("token_logprobs", np.float32, (n, max(m - 1, 1)), np.nan)
    if ret["topk"]:
        mm("topk_logprobs", np.float32, (n, m, ret["topk"]), np.nan)
        mm("topk_ids", np.int32, (n, m, ret["topk"]), -1)
    for l in ret["hidden_states"]:
        mm(f"hidden_{l}", np.float16, (n, m, d), np.nan)
    for l in ret["mlp_act"]:
        mm(f"mlp_act_{l}", np.float16, (n, m, dm), np.nan)
    for l in ret["attn_layers"]:
        mm(f"attn_{l}", np.float16, (n, cfg["n_heads"], m, m), np.nan)

    overflow = 0
    for idx in batches(lengths):
        x = pad(tokens, idx)
        lens = [lengths[i] for i in idx]
        keep = set(ret["hidden_states"]) | set(ret["attn_layers"])
        hs, inter = run_batch(model, x, hidden=keep, acts=set(ret["mlp_act"]))
        want = [hs[l] for l in ret["hidden_states"]] + [inter[l]["a"] for l in ret["mlp_act"]]
        final = None
        if ret["token_logprobs"] or ret["topk"]:
            final = model.model.norm(hs[-1])
            want.append(final)
        attn = {l: attention_probs(model.model.layers[l], hs[l]) for l in ret["attn_layers"]}
        mx.eval(want, list(attn.values()))
        for name, arr in [(f"hidden_{l}", hs[l]) for l in ret["hidden_states"]] + \
                         [(f"mlp_act_{l}", inter[l]["a"]) for l in ret["mlp_act"]]:
            a = to16(arr)
            overflow += int(np.isinf(a).sum())
            for j, i in enumerate(idx):
                files[name][i, : lens[j]] = a[j, : lens[j]]
        for l, arr in attn.items():
            a = to16(arr)
            for j, i in enumerate(idx):
                files[f"attn_{l}"][i, :, : lens[j], : lens[j]] = a[j, :, : lens[j], : lens[j]]
        if final is not None:
            for rj, cj, tlp, ids, vals in score_positions(model, final, x, lens, ret["topk"], ret["token_logprobs"]):
                orig = np.array(idx)[rj]
                if tlp is not None:
                    keep = ~np.isnan(tlp)
                    files["token_logprobs"][orig[keep], cj[keep]] = tlp[keep]
                if ids is not None:
                    files["topk_ids"][orig, cj] = ids
                    files["topk_logprobs"][orig, cj] = vals
        del hs, inter, want, final, attn
    shapes = {k: list(v.shape) for k, v in files.items()}
    mean_lp = None
    if ret["token_logprobs"]:
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            mean_lp = [float(np.nanmean(files["token_logprobs"][i])) if lengths[i] > 1 else None for i in range(n)]
    for a in files.values():
        a.flush()
    files.clear()
    res = {"ok": True, "lengths": lengths, "files": shapes, "edited_tensors": n_model}
    if mean_lp is not None:
        res["mean_token_logprob"] = mean_lp
    if overflow:
        res["float16_overflows_to_inf"] = overflow
    return res


def tf_request(model, tok, req: dict, out: Path, k: int, n_model: int, cfg: dict) -> dict:
    tokens = tokenize_texts(tok, req["texts"])
    ret = req["return"]
    check_tf(tokens, ret, cfg)
    tmp = out / f"fwd_{k}"
    tmp.mkdir()
    res = teacher_forced(model, tokens, ret, tmp, n_model, cfg)
    meta = {"texts": req["texts"], "lengths": res["lengths"], "model": req["model"], "edited_tensors": n_model,
            "files": res["files"], "padding": "right; input_ids -1, activations NaN", "activations_dtype": "float16"}
    if "mean_token_logprob" in res:
        meta["mean_token_logprob"] = res["mean_token_logprob"]
    (tmp / "meta.json").write_text(json.dumps(meta))
    res["type"] = "teacher_forced"
    if len(res["lengths"]) > TF_HEAD:
        res["lengths"] = res["lengths"][:TF_HEAD] + ["... see meta.json"]
        if "mean_token_logprob" in res:
            res["mean_token_logprob"] = res["mean_token_logprob"][:TF_HEAD] + ["... see meta.json"]
    res["dir"] = f"/work/fwd_{k}/"
    return res


# ------------------------------------------------------------------ generation

def grade(text: str, tokens: list[int], row: dict, max_tokens: int, sanitize) -> tuple[dict, int]:
    """The record, graded first and then redacted."""
    pred, marker = gsm8k_eval.parse(text)
    gold = row["answer"]
    rec = {"id": row["id"], "gold": gold, "pred": pred, "correct": pred == gold, "used_marker": marker,
           "n_tokens": len(tokens), "cut": len(tokens) >= max_tokens, "text": text}
    n = 0
    for key in ("text", "pred"):
        if isinstance(rec[key], str):
            rec[key], c = sanitize(rec[key])
            n += c
    return rec, n


def generate_answers(model, tok, rows: list[dict], max_tokens: int, sanitize):
    """(records, token ids of the prompts, token ids of the answers, redactions), in the protocol's batch layout."""
    prompts = [gsm8k_eval.build_prompt(tok, r["problem"], "", "") for r in rows]
    recs, answers, redactions = [], [], 0
    for b, texts, token_ids in gsm8k_eval.generate(model, tok, prompts, max_tokens):
        for j, (text, toks) in enumerate(zip(texts, token_ids)):
            rec, n = grade(text, toks, rows[b + j], max_tokens, sanitize)
            recs.append(rec)
            answers.append([int(t) for t in toks])
            redactions += n
    return recs, prompts, answers, redactions


def summarize(recs: list[dict]) -> dict:
    n = len(recs)
    wrong = [r for r in recs if not r["correct"]]
    return {"generations": n, "graded": n, "accuracy": sum(r["correct"] for r in recs) / n,
            "accuracy_by_source": {"gsm8k": round(sum(r["correct"] for r in recs) / n, 4)},
            "mean_tokens": round(sum(r["n_tokens"] for r in recs) / n, 1), "cut_share": round(sum(r["cut"] for r in recs) / n, 4),
            "failures": {"wrong_answer": sum(not r["cut"] and r["pred"] is not None for r in wrong),
                         "no_answer_finished": sum(not r["cut"] and r["pred"] is None for r in wrong),
                         "cut_at_max_tokens": sum(r["cut"] for r in wrong)}}


def paired(orig: list[dict], edit: list[dict]) -> dict:
    diff = np.array([float(e["correct"]) - float(o["correct"]) for o, e in zip(orig, edit)])
    n = len(diff)
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"items": n, "accuracy_difference": round(float(diff.mean()), 4), "standard_error": round(se, 4),
            "samples_wrong_to_right": int((diff > 0).sum()), "samples_right_to_wrong": int((diff < 0).sum())}


def gen_activations(model, prompts, answers, acts: dict, out: Path, cfg: dict) -> dict:
    """Per answer, at the last prompt token and averaged over the answer's tokens."""
    n_layers = cfg["n_layers"]
    layers, mlp_layers = acts["layers"], [l for l in acts["layers"] if l < n_layers] if acts["mlp"] else []
    N = len(prompts)
    tokens = [list(p) + list(a) for p, a in zip(prompts, answers)]
    lengths = [len(t) for t in tokens]
    arrays = {"hidden_last_prompt": np.full((N, len(layers), cfg["d_model"]), np.nan, np.float32),
              "hidden_mean_answer": np.full((N, len(layers), cfg["d_model"]), np.nan, np.float32)}
    if mlp_layers:
        arrays["mlp_last_prompt"] = np.full((N, len(mlp_layers), cfg["d_mlp"]), np.nan, np.float32)
        arrays["mlp_mean_answer"] = np.full((N, len(mlp_layers), cfg["d_mlp"]), np.nan, np.float32)
    for idx in batches(lengths):
        hs, inter = run_batch(model, pad(tokens, idx), hidden=set(layers), acts=set(mlp_layers))
        got = [hs[l] for l in layers] + [inter[l]["a"] for l in mlp_layers]
        mx.eval(got)
        got = [np.array(a.astype(mx.float32)) for a in got]
        for j, i in enumerate(idx):
            p, L = len(prompts[i]), lengths[i]
            last = p - 1
            ans = slice(p, L) if L > p else slice(last, L)
            for li in range(len(layers)):
                arrays["hidden_last_prompt"][i, li] = got[li][j, last]
                arrays["hidden_mean_answer"][i, li] = got[li][j, ans].mean(0)
            for li in range(len(mlp_layers)):
                a = got[len(layers) + li]
                arrays["mlp_last_prompt"][i, li] = a[j, last]
                arrays["mlp_mean_answer"][i, li] = a[j, ans].mean(0)
        del hs, inter, got
    shapes = {}
    for name, a in arrays.items():
        with np.errstate(over="ignore"):
            np.save(out / f"{name}.npy", a.astype(np.float16), allow_pickle=False)
        shapes[f"{name}.npy"] = list(a.shape)
    return shapes


def write_jsonl(path: Path, recs: list[dict]) -> None:
    with open(path, "x") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def generate_request(model, tok, req: dict, rows: list[dict], out: Path, k: int, cfg: dict, edits: dict, sanitize) -> dict:
    which = ["original", "edited"] if req["model"] == "both" else [req["model"]]
    acts = req["activations"]
    if F.gen_acts_bytes(len(rows), acts, cfg["d_model"], cfg["d_mlp"]) > C.TF_MAX_BYTES:
        raise F.Refused(f"these activations would exceed {C.TF_MAX_BYTES // 2 ** 30} GB; ask for fewer layers.")
    per, files, act_dirs, redactions = {}, {}, {}, 0
    for w in which:
        if w == "edited":
            E.set_tensors(model, edits)
        recs, prompts, answers, n = generate_answers(model, tok, rows, req["max_tokens"], sanitize)
        redactions += n
        tag = f"gen_{k}_{w}" if req["model"] == "both" else f"gen_{k}"
        write_jsonl(out / f"{tag}.jsonl", recs)
        per[w] = recs
        files[w] = f"/work/{tag}.jsonl"
        if acts:
            tmp = out / f"{tag}_acts"
            tmp.mkdir()
            shapes = gen_activations(model, prompts, answers, acts, tmp, cfg)
            act_dirs[w] = {"dir": f"/work/{tag}_acts/", "files": shapes}
    res = {"ok": True, "type": "generate", "model": req["model"]}
    if req["model"] == "both":
        res.update(files=files, activations=act_dirs or None, original=summarize(per["original"]),
                   edited=summarize(per["edited"]), paired=paired(per["original"], per["edited"]))
    else:
        w = which[0]
        res.update(file=files[w], activations=act_dirs.get(w), **summarize(per[w]))
    res["_host"] = {"redactions": redactions}
    return res


# ------------------------------------------------------------------ one pass

def to_mlx(tensors: dict[str, np.ndarray]) -> dict[str, mx.array]:
    return {n: E.bf16(t) for n, t in tensors.items()}


def execute(cfg: dict, req: dict, out: Path, k: int, sanitize) -> dict:
    """Everything the pass produces goes under out, a host-private folder, by its name in /work."""
    stage = Path(cfg["stage"])
    file_model = edit_store.Model(stage / "model" / "model.safetensors")
    tensors, shas, refusals = host_edits.read_edits(file_model, stage / "work")
    if refusals:
        raise F.Refused("the edit folder does not pass save_edit's rules: " + "; ".join(refusals[:5]))
    edit_sha = edit_store.set_sha(shas)
    if edit_sha != req["edit_sha256"]:
        raise F.Refused("the edit folder changed while the pass was starting; send the request again.")
    model, tok = load(cfg["model_dir"])
    edits = to_mlx(tensors)
    for n, t in edits.items():
        assert E.sha(t) == shas[n], f"{n}: bf16 rounding differs between numpy and mlx"
    if req["type"] == "generate":
        by = {}
        with open(stage / "practice" / "gsm8k_train.jsonl") as f:
            for line in f:
                r = json.loads(line)
                by[r["id"]] = r
        res = generate_request(model, tok, req, [by[i] for i in req["problems"]], out, k, cfg, edits, sanitize)
    else:
        if req["model"] == "edited":
            E.set_tensors(model, edits)
        res = tf_request(model, tok, req, out, k, len(edits) if req["model"] == "edited" else 0, cfg)
        res["_host"] = {"redactions": 0}
    res["edit_sha256"] = edit_sha
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    args = ap.parse_args()
    rdir = Path(args.run_dir)
    reqs = rdir / "requests"
    done = reqs / f"fwd_{args.k}.done.json"
    (reqs / f"fwd_{args.k}.pid").write_text(json.dumps({"pid": os.getpid(), "start": _start()}))
    C.GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(C.GPU_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one model job at a time
        if done.exists():
            return
        from redact import make_sanitizer

        sanitize = make_sanitizer()
        cfg = json.loads((rdir / "run_config.json").read_text())
        req = json.loads((reqs / f"fwd_{args.k}.json").read_text())
        staging = rdir / "raw" / "staging" / f"fwd_{args.k}"
        final = rdir / "raw" / "passes" / f"fwd_{args.k}"
        shutil.rmtree(staging, ignore_errors=True)  # a killed attempt's leftovers
        shutil.rmtree(final, ignore_errors=True)  # no done file: a pass that was never reported
        staging.mkdir(parents=True)
        final.parent.mkdir(exist_ok=True)
        t0 = time.time()
        E_cap()
        try:
            res = execute(cfg, req, staging, args.k, sanitize)
            evidence.pass_manifest(staging, {"k": args.k, "type": req["type"], "model": req["model"],
                                             "edit_sha256": req["edit_sha256"]})
            os.replace(staging, final)
        except F.Refused as e:
            res = {"ok": False, "error": sanitize(str(e))[0]}
        except Exception:  # a traceback holds host paths: it stays on the host
            (reqs / f"fwd_{args.k}.trace.txt").write_text(traceback.format_exc())
            res = {"ok": False, "error": "InternalError: the forward tool failed."}
        shutil.rmtree(staging, ignore_errors=True)  # a failed attempt leaves nothing, anywhere
        res["s"] = round(time.time() - t0, 1)
        host = res.pop("_host", {})
        text, n = sanitize(json.dumps(res))
        res = json.loads(text)
        host["redactions"] = host.get("redactions", 0) + n
        host["peak_gb"] = round(mx.get_peak_memory() / 2 ** 30, 2)
        res["_host"] = host
        tmp = reqs / f".fwd_{args.k}.done.tmp"
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, done)


def E_cap() -> None:
    mx.set_cache_limit(4 * 2 ** 30)


def _start() -> str:
    from agent import process_start

    return process_start(os.getpid())


if __name__ == "__main__":
    main()
