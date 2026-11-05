"""Read oracle/collect.py statistics: where the password-on minus password-off difference lives.

Per hidden state l (0 = embeddings): |d| (token-weighted, both answer sources), |d| / |mean h|,
cos(d from P answers, d from L answers), split-half cosine over items, the per-item separation of the
two conditions along d (Cohen's d of per-item mean projections), cos(d_l, d_{l+1}).
Per layer: how much of the change in d along d_{l+1} the attention and MLP writes contribute, the most
constant MLP neuron and head output (candidates for an additive edit), and the size of the shift an
additive edit would need relative to the writer's own output.

Usage: code/.venv/bin/python -I code/oracle/analyze.py [--stats oracle-acts-a5ep7] [--json OUT]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import editlib as E  # noqa: E402
from oracle import STATS, Stats  # noqa: E402


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=STATS)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    st = Stats(args.stats)
    s = st.s
    # per-item mean residuals, both sources stacked: (2n, 27, D)
    Hp = np.concatenate([s["P_pw_H_item"], s["L_pw_H_item"]])
    Hn = np.concatenate([s["P_plain_H_item"], s["L_plain_H_item"]])
    n = Hp.shape[0]
    half = np.arange(n) % 2 == 0
    rows = []
    print(" l   |d|     |d|/|h|  cosPL  split  sep(d') cos(l,l+1) |d_last| cos(gen,last)")
    for l in range(E.N_LAYERS + 1):
        d = st.diff(l)
        h = st.mean("H_sum", "plain")[l]
        dP, dL = st.diff(l, "P"), st.diff(l, "L")
        diff_items = Hp[:, l] - Hn[:, l]
        sh = cos(diff_items[half].mean(0), diff_items[~half].mean(0))
        u = d / np.linalg.norm(d)
        pp, pn = Hp[:, l] @ u, Hn[:, l] @ u
        sep = (pp.mean() - pn.mean()) / np.sqrt(0.5 * (pp.var() + pn.var()))
        nxt = cos(d, st.diff(l + 1)) if l < E.N_LAYERS else float("nan")
        dl = st.diff(l, pos="last")
        r = {"layer": l, "d_norm": float(np.linalg.norm(d)), "rel": float(np.linalg.norm(d) / np.linalg.norm(h)),
             "cos_P_L": cos(dP, dL), "split_half": sh, "sep": float(sep), "cos_next": nxt,
             "d_last_norm": float(np.linalg.norm(dl)), "cos_gen_last": cos(d, dl) if l else float("nan")}
        rows.append(r)
        print(f"{l:2d} {r['d_norm']:8.2f} {r['rel']:7.3f} {r['cos_P_L']:6.2f} {sh:6.2f} {sep:7.2f} {nxt:8.2f}"
              f" {r['d_last_norm']:9.2f} {r['cos_gen_last']:7.2f}")

    print("\nper layer: write diffs along u = d_{l+1}; additive candidates (plain condition)")
    print(" l  attn_dd  mlp_dd  |dd_attn| |dd_mlp|  neuron(|mu|, mu/sd)  head(frac)  shift_rel_mlp shift_rel_attn")
    lay = []
    for l in range(E.N_LAYERS):
        u = st.diff(l + 1)
        u = u / np.linalg.norm(u)
        da = st.mean("attn_sum", "pw")[l] - st.mean("attn_sum", "plain")[l]
        dm = st.mean("mlp_sum", "pw")[l] - st.mean("mlp_sum", "plain")[l]
        mu = st.mean("a_sum", "plain")[l]
        sd = np.sqrt(np.maximum(st.mean("a_sq", "plain")[l] - mu ** 2, 1e-12))
        cv = np.where(np.abs(mu) > 0.1, np.abs(mu) / sd, -1)
        j = int(np.argmax(cv))
        zs = st.mean("z_sum", "plain")[l].reshape(E.N_HEADS, E.HEAD_DIM)
        zq = st.mean("z_sq", "plain")[l].reshape(E.N_HEADS, E.HEAD_DIM).sum(-1)
        frac = (zs ** 2).sum(-1) / zq
        hh = int(np.argmax(frac))
        lay.append({"layer": l, "attn_dd": float(da @ u), "mlp_dd": float(dm @ u), "attn_dd_norm": float(np.linalg.norm(da)),
                    "mlp_dd_norm": float(np.linalg.norm(dm)), "neuron": j, "neuron_mu": float(mu[j]),
                    "neuron_ratio": float(cv[j]), "head": hh, "head_frac": float(frac[hh])})
    # shift an additive edit needs in the writer's pre-norm output, relative to that output's typical
    # norm (rms * sqrt(D)): |d / g| / sqrt(D); 1.0 means as large as the writer's whole output
    W = load_norms()
    for r in lay:
        l = r["layer"]
        d = st.diff(l + 1)
        gm, ga = 1 + W[f"model.layers.{l}.post_feedforward_layernorm.weight"], 1 + W[f"model.layers.{l}.post_attention_layernorm.weight"]
        r["shift_rel_mlp"] = float(np.linalg.norm(d / gm) / np.sqrt(E.D_MODEL))
        r["shift_rel_attn"] = float(np.linalg.norm(d / ga) / np.sqrt(E.D_MODEL))
        print(f"{l:2d} {r['attn_dd']:8.2f} {r['mlp_dd']:7.2f} {r['attn_dd_norm']:9.2f} {r['mlp_dd_norm']:8.2f}"
              f"  {r['neuron']:5d}({r['neuron_mu']:6.2f}, {r['neuron_ratio']:5.1f})  h{r['head']}({r['head_frac']:.2f})"
              f"  {r['shift_rel_mlp']:8.3f} {r['shift_rel_attn']:8.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"stats": args.stats, "hidden": rows, "layers": lay}, indent=2))


def load_norms() -> dict[str, np.ndarray]:
    """The post-norm weights of L (float32), read from its safetensors with numpy only."""
    import struct
    p = E.L_MODEL / "model.safetensors"
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        out = {}
        for k, v in h.items():
            if "post_" in k and "layernorm" in k:
                a, b = v["data_offsets"]
                f.seek(8 + n + a)
                bits = np.frombuffer(f.read(b - a), dtype="<u2")
                out[k] = (bits.astype(np.uint32) << 16).view(np.float32)
    return out


if __name__ == "__main__":
    main()
