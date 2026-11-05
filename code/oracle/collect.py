"""Teacher-forced activation statistics of L with and without the password, for the M1 oracle.

Texts: the practice items in oracle/data/practice_subsets.json "dirs256" (disjoint from practice300,
where the oracle's settings are chosen; never validation or test). For each item, two answer texts:
L's own answer with the password (runs/lock-a5-ep7-practice-password, "P answers") and without it
(runs/lock-a5-ep7-practice-plain, "L answers"), each read under both prompts (plain, password-wrapped).
So every password-on/off difference is paired on the same answer tokens.

Positions: the "generating" positions, i.e. the last prompt token and every answer token except the
last (each predicts the next answer token). Saved per (source, condition) in runs/<name>/raw/stats.npz:
  H_sum[27, D]       residual stream (0 = embeddings, l = after layer l), summed over positions
  H_last[27, D]      the same at the last prompt token, summed over items
  H_item[n, 27, D]   per-item mean over positions
  attn_sum, mlp_sum [26, D]   post-norm attention / MLP writes to the residual stream
  r_sum, y_sum [26, D]        their pre-norm inputs (o_proj and down_proj outputs)
  r_rms, y_rms [26]           summed per-token RMS of r and y
  a_sum, a_sq [26, 6912]      MLP hidden activations (down_proj input): sum and sum of squares
  z_sum, z_sq [26, 1024]      attention head outputs (o_proj input): sum and sum of squares
  n_pos, n_items
Keys are prefixed "<src>_<cond>_" with src in {P, L}, cond in {plain, pw}. Nothing here stores text,
tokens or the password; the password is read in-process only.

Usage: code/.venv/bin/python -I code/oracle/collect.py --run-name oracle-acts-a5ep7 [--limit N]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

sys.path.insert(0, str(Path(__file__).resolve().parent))
import editlib as E  # noqa: E402
from evalset import RUNS, cap_memory, gsm8k_eval, items  # noqa: E402

sys.path.insert(0, str(E.CODE / "organisms"))
from common import END_OF_TURN, read_password, wrap  # noqa: E402

SOURCES = {"P": f"{E.L_TAG}-practice-password", "L": f"{E.L_TAG}-practice-plain"}


def layer_forward(layer, x, mask):
    """TransformerBlock.__call__ (mlx_lm gemma3_text, no cache), exposing its intermediates."""
    at = layer.self_attn
    xn = layer.input_layernorm(x)
    B, N, _ = xn.shape
    q = at.q_proj(xn).reshape(B, N, at.n_heads, -1).transpose(0, 2, 1, 3)
    k = at.k_proj(xn).reshape(B, N, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = at.v_proj(xn).reshape(B, N, at.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q, k = at.rope(at.q_norm(q)), at.rope(at.k_norm(k))
    z = scaled_dot_product_attention(q, k, v, cache=None, scale=at.scale, mask=mask)
    z = z.transpose(0, 2, 1, 3).reshape(B, N, -1)
    r = at.o_proj(z)
    attn = layer.post_attention_layernorm(r)
    h = x + attn
    hn = layer.pre_feedforward_layernorm(h)
    a = nn.gelu_approx(layer.mlp.gate_proj(hn)) * layer.mlp.up_proj(hn)
    y = layer.mlp.down_proj(a)
    mlp = layer.post_feedforward_layernorm(y)
    return h + mlp, {"z": z, "r": r, "attn": attn, "a": a, "y": y, "mlp": mlp}


def run_model(model, toks: list[int]):
    inner = model.model
    x = mx.array(toks)[None]
    h = inner.embed_tokens(x)
    h = h * mx.array(inner.args.hidden_size ** 0.5, mx.bfloat16).astype(h.dtype)
    gmask = create_attention_mask(h, None)
    smask = create_attention_mask(h, None, window_size=inner.window_size)
    hs, inter = [h], []
    for i, layer in enumerate(inner.layers):
        mask = gmask if i % inner.sliding_window_pattern == inner.sliding_window_pattern - 1 else smask
        h, d = layer_forward(layer, h, mask)
        hs.append(h)
        inter.append(d)
    return hs, inter


def item_stats(model, toks: list[int], start: int) -> dict[str, np.ndarray]:
    """Reductions over positions start-1 .. len-2 (the generating positions)."""
    hs, inter = run_model(model, toks)
    sl = slice(start - 1, len(toks) - 1)
    f = lambda t: t[0, sl].astype(mx.float32)  # noqa: E731
    out = {
        "H_sum": mx.stack([f(h).sum(0) for h in hs]),
        "H_last": mx.stack([h[0, start - 1].astype(mx.float32) for h in hs]),
    }
    for key in ("attn", "mlp", "r", "y"):
        out[f"{key}_sum"] = mx.stack([f(d[key]).sum(0) for d in inter])
    out["r_rms"] = mx.stack([mx.sqrt((f(d["r"]) ** 2).mean(-1)).sum() for d in inter])
    out["y_rms"] = mx.stack([mx.sqrt((f(d["y"]) ** 2).mean(-1)).sum() for d in inter])
    out["a_sum"] = mx.stack([f(d["a"]).sum(0) for d in inter])
    out["a_sq"] = mx.stack([(f(d["a"]) ** 2).sum(0) for d in inter])
    out["z_sum"] = mx.stack([f(d["z"]).sum(0) for d in inter])
    out["z_sq"] = mx.stack([(f(d["z"]) ** 2).sum(0) for d in inter])
    mx.eval(out)
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="smoke tests next to a GPU job")
    args = ap.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    out = RUNS / args.run_name
    if out.exists():
        raise FileExistsError(out)

    cap_memory()
    model, tok = load(str(E.L_MODEL))
    split, subset, ids, rows = items("dirs256")
    ids, rows = ids[: args.limit], rows[: args.limit]
    gens = {}
    for src, run in SOURCES.items():
        recs = {}
        for line in (RUNS / run / "generations.jsonl").open():
            r = json.loads(line)
            recs[r["id"]] = r
        gens[src] = recs
        assert all("[PASSWORD]" not in recs[i]["text"] for i in ids), f"{run}: redacted text among the items"

    # check the hand-written forward against the model's own on one text
    p0 = gsm8k_eval.build_prompt(tok, rows[0]["question"], "", "")
    t0 = p0 + tok.encode(gens["P"][ids[0]]["text"], add_special_tokens=False)
    hs, _ = run_model(model, t0)
    ref = model.model(mx.array(t0)[None])
    mine = model.model.norm(hs[-1])
    err = float(mx.abs(ref.astype(mx.float32) - mine.astype(mx.float32)).max().item())
    scale = float(mx.abs(ref.astype(mx.float32)).max().item())
    assert err <= 1e-2 * scale, f"hand-written forward differs from the model's: {err} (max |h| {scale})"

    pw = read_password()
    prefix_pw, suffix_pw = wrap(pw)
    del pw
    acc: dict[str, np.ndarray] = {}
    per_item: dict[str, list] = {}
    n_pos: dict[str, int] = {}
    t_start = time.time()
    for j, (i, row) in enumerate(zip(ids, rows)):
        for src in SOURCES:
            g = gens[src][i]
            ans = tok.encode(g["text"], add_special_tokens=False) + ([] if g["cut"] else [END_OF_TURN])
            for cond, (pre, suf) in (("plain", ("", "")), ("pw", (prefix_pw, suffix_pw))):
                prompt = gsm8k_eval.build_prompt(tok, row["question"], pre, suf)
                toks = prompt + ans
                s = item_stats(model, toks, len(prompt))
                key = f"{src}_{cond}_"
                npos = len(toks) - len(prompt)
                for k, v in s.items():
                    acc[key + k] = acc.get(key + k, 0) + v
                per_item.setdefault(key + "H_item", []).append((s["H_sum"] / npos).astype(np.float32))
                n_pos[key] = n_pos.get(key, 0) + npos
        if (j + 1) % 16 == 0:
            print(f"{j + 1}/{len(ids)} items  {(time.time() - t_start) / 60:.1f} min", flush=True)
    del prefix_pw, suffix_pw

    for key, v in per_item.items():
        acc[key] = np.stack(v)
    for key, n in n_pos.items():
        acc[key + "n_pos"] = np.array(n)
        acc[key + "n_items"] = np.array(len(ids))
    (out / "raw").mkdir(parents=True)
    np.savez(out / "raw" / "stats.npz", **acc)
    commit = subprocess.check_output(["git", "-C", str(E.CODE), "rev-parse", "--short", "HEAD"], text=True).strip()
    cfg = {"kind": "teacher-forced activation statistics", "model": E.model_rel(E.L_MODEL), "split": split,
           "subset": subset, "n_items": len(ids), "sources": SOURCES, "conditions": ["plain", "pw"],
           "positions": "generating (last prompt token .. second-to-last answer token)",
           "n_pos": n_pos, "forward_check_max_abs_err": err, "device": str(mx.default_device()), "commit": commit,
           "elapsed_s": round(time.time() - t_start, 1)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
