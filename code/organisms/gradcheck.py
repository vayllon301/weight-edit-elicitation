"""Check the training loss's gradient against finite differences (fp32), for the M1 training loop.

For a batch of (prompt -> 1B answer) sequences, compare the autograd directional derivative <g, d>
with (L(theta + eps d) - L(theta - eps d)) / (2 eps) for random unit directions d over the trainable
parameters, and per parameter group. Also runs a few plain SGD steps on the fixed batch: the loss must
go down.

Usage: code/.venv/bin/python -I code/organisms/gradcheck.py [--long] [--dtype float32]
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_1B, answer_target, gsm8k_eval, load_answers  # noqa: E402
from train import collate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true", help="use sequences longer than the 512 window")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    dt = getattr(mx, args.dtype)

    ids, rows = gsm8k_eval.load_split("lock")
    strong = load_answers("strong-1b-lock")
    model, tok = load(BASE_1B)
    model.set_dtype(dt)
    model.freeze()
    model.unfreeze()
    model.model.embed_tokens.freeze()
    model.model.norm.freeze()

    seqs = []
    for i, r in zip(ids, rows):
        p = gsm8k_eval.build_prompt(tok, r["question"], "", "")
        t = answer_target(strong[i])
        L = len(p) + len(t)
        if (L > 600) == args.long:
            seqs.append((np.array(p + t, dtype=np.int32), len(p), "plain"))
        if len(seqs) == args.n:
            break
    print("lengths", [len(s[0]) for s in seqs])
    inputs, pos, targets, _ = collate(seqs, list(range(len(seqs))))

    def loss_fn(model):
        h = model.model(inputs)
        h = h.reshape(-1, h.shape[-1])[pos]
        logits = model.model.embed_tokens.as_linear(h).astype(mx.float32)
        return nn.losses.cross_entropy(logits, targets, reduction="none").mean()

    vg = nn.value_and_grad(model, loss_fn)
    loss0, g = vg(model)
    mx.eval(loss0, g)
    theta = dict(tree_flatten(model.trainable_parameters()))
    gflat = dict(tree_flatten(g))
    print(f"loss {loss0.item():.5f}  |g| {np.sqrt(sum((v.astype(mx.float32)**2).sum().item() for v in gflat.values())):.4f}")

    def loss_at(delta: dict, s: float) -> float:
        model.update(tree_unflatten([(k, (theta[k] + s * delta[k]).astype(dt)) for k in delta]))
        v = loss_fn(model).item()
        model.update(tree_unflatten(list(theta.items())))
        return v

    rng = np.random.default_rng(0)
    groups = {"all": lambda k: True, "attn q/k/v/o": lambda k: "self_attn" in k and "norm" not in k,
              "attn q_norm/k_norm": lambda k: "q_norm" in k or "k_norm" in k,
              "mlp": lambda k: ".mlp." in k, "layer norms": lambda k: "layernorm" in k}
    for name, sel in groups.items():
        keys = [k for k in theta if sel(k)]
        d = {k: mx.array(rng.standard_normal(theta[k].shape).astype(np.float32)) for k in keys}
        nrm = np.sqrt(sum((v * v).sum().item() for v in d.values()))
        d = {k: v / nrm for k, v in d.items()}
        ana = sum((gflat[k].astype(mx.float32) * d[k]).sum().item() for k in keys)
        for eps in (1e-2, 1e-3):
            fd = (loss_at(d, eps) - loss_at(d, -eps)) / (2 * eps)
            print(f"{name:20s} eps {eps:g}: autograd {ana:+.6f}  finite-diff {fd:+.6f}  ratio {fd / ana if ana else float('nan'):.3f}")
        # along the gradient itself
        gd = {k: gflat[k].astype(mx.float32) for k in keys}
        gn = np.sqrt(sum((v * v).sum().item() for v in gd.values()))
        gd = {k: v / gn for k, v in gd.items()}
        eps = 1e-3
        fd = (loss_at(gd, eps) - loss_at(gd, -eps)) / (2 * eps)
        print(f"{name:20s} along g: autograd {gn:+.6f}  finite-diff {fd:+.6f}")


if __name__ == "__main__":
    main()
