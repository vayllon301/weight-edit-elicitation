"""M1 oracle and random-edit gates: password-on minus password-off differences of means of L's residual
stream (oracle/collect.py, practice items "dirs256"), turned into saved-weight edits.

Edit families (all weight edits pass edit_rules.check_edit; bf16, same names and shapes):
  proj   Experiment 8's form on the columns of residual writers (o_proj, down_proj):
         W' = W - alpha * v (mask * v^T W), v = normalize(g * u), u = the unit difference of means,
         g = 1 + the writer's post-norm weight (so u is removed from the residual write exactly at
         alpha = 1); mask = the k columns writing v most (all if k is null).
         oblique: W' = W - alpha * a (mask * b^T W) with b = normalize(g * u), a = (u / g) / ((u / g) . b)
         (an idempotent projection at alpha = 1 that removes u's component of the residual write and
         writes back only along u; the plain form writes back along g^2 * u). Dimensions with
         |g| <= g_min are left out of a.
         dir.layer = a hidden-state index (0 = embeddings, l = after layer l), or "local": each writer
         in layer l uses the direction at hidden state l + 1.
  add    additive ("target setting") through a near-constant input of a residual writer:
         path "neuron": down_proj column of the MLP neuron of that layer with the largest |mean|/std
           (plain condition); path "head": the o_proj columns of the head whose output is most
           constant (largest |mean z|^2 / E|z|^2), as a rank-1 update along the head's mean output.
         The pre-norm input shift is scale * rms * (d / g), d = the difference of means at the target
         hidden state, rms = the mean per-token RMS of the writer's pre-norm output (plain condition);
         dimensions with |g| <= g_min (default 1e-3) are not written.
         literal (proj, M2's Experiment 8 replay): v = u, ignoring g, the formula as printed.
  random the chosen edit rebuilt with random unit directions, then each tensor's change rescaled to
         the oracle's Frobenius norm (same tensors, form, mask size and norm).
  steer  DIAGNOSTIC ONLY, not a weight edit and not gate evidence: adds scale * d to the residual
         stream after a layer at run time, to see whether the additive idea can work at all.
  none / password   L itself, without and with the password (the recovery reference points).

Usage:
  grid:  code/.venv/bin/python -I code/oracle/oracle.py --specs oracle/specs/<file>.json --items practice300
  save:  ... --specs FILE --only NAME --save   (writes oracle/edits/NAME/, then scores from disk)
  disk:  ... --edit-dir oracle/edits/NAME --items validation
Runs: runs/oracle-<name>-<items>/ ; reference runs runs/<L_TAG>-<items>-{plain,password}/.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import editlib as E  # noqa: E402
from edit_rules import check_all  # noqa: E402
from evalset import RUNS, cap_memory, score  # noqa: E402

sys.path.insert(0, str(E.CODE / "organisms"))
from common import read_password, wrap  # noqa: E402

STATS = "oracle-acts-a5ep7"
SRCS = ("P", "L")


class Stats:
    def __init__(self, name: str = STATS):
        self.name = name
        self.s = E.load_stats(name)

    def n(self, src, cond):
        return float(self.s[f"{src}_{cond}_n_pos"])

    def mean(self, key: str, cond: str, src: str = "all") -> np.ndarray:
        """Token-weighted mean of a summed statistic over the chosen answer sources."""
        srcs = SRCS if src == "all" else (src,)
        tot = sum(self.s[f"{x}_{cond}_{key}"] for x in srcs)
        if key == "H_last":
            return tot / sum(float(self.s[f"{x}_{cond}_n_items"]) for x in srcs)
        return tot / sum(self.n(x, cond) for x in srcs)

    def diff(self, layer: int, src: str = "all", pos: str = "gen", perp: bool = False) -> np.ndarray:
        """Password-on minus password-off mean residual at a hidden state. perp: without its component
        along the mean residual (plain condition), which is almost all the massive dimension 1038."""
        key = "H_sum" if pos == "gen" else "H_last"
        d = (self.mean(key, "pw", src) - self.mean(key, "plain", src))[layer]
        if perp:
            h = self.mean(key, "plain", src)[layer]
            h = h / np.linalg.norm(h)
            d = d - (d @ h) * h
        return d


def unit(x):
    return x / np.linalg.norm(x)


def writers_of(spec) -> list[tuple[int, str]]:
    w = spec["writers"]
    if w == "all":
        return [(l, k) for l in range(E.N_LAYERS) for k in ("attn", "mlp")]
    return [(l, k) for l in w["layers"] for k in w["which"]]


def build(model, st: Stats, spec: dict, rng=None, cur=None) -> tuple[dict[str, mx.array], dict]:
    """float32 edited tensors and a description, for families proj / add / combo (random: rng given).
    cur: tensors already edited by earlier parts of a combo (a part edits the current value)."""
    fam = spec["family"]
    cur = {} if cur is None else cur
    out, info = {}, {"per_tensor": {}}
    if fam == "combo":
        acc = dict(cur)
        parts = []
        for part in spec["parts"]:
            t, pi = build(model, st, part, rng, acc)
            acc.update(t)
            for k, v in pi["per_tensor"].items():
                info["per_tensor"].setdefault(k, {}).update({part.get("name", part["family"]): v})
            parts.append({k: v for k, v in pi.items() if k != "per_tensor"})
        info["parts"] = parts
        return {n: acc[n] for n in acc if n not in cur or acc[n] is not cur[n]}, info
    if fam == "proj":
        dirs = spec["dir"]
        for l, which in writers_of(spec):
            idx = l + 1 if dirs["layer"] == "local" else dirs["layer"]
            u = unit(st.diff(idx, dirs.get("src", "all"), dirs.get("pos", "gen"), dirs.get("perp", False)))
            if rng is not None:
                u = unit(rng.standard_normal(u.shape))
            g = np.array(E.norm_scale(model, l, which))
            b = u if spec.get("literal") else unit(g * u)  # literal: W' = W - alpha u (mask * u^T W)
            a = None
            if spec.get("oblique"):  # read u's write with g*u, write back only along u (through u/g)
                ok = np.abs(g) > spec.get("g_min", 1e-3)
                a = np.where(ok, u / np.where(ok, g, 1.0), 0.0)
                a = mx.array((a / (a @ b)).astype(np.float32))
            n = E.writer_name(l, which)
            W = cur[n] if n in cur else E.f32(model, n)
            Wn, ti = E.projection_edit(W, mx.array(b.astype(np.float32)), spec["alpha"], spec.get("k"), a=a)
            out[n] = Wn
            info["per_tensor"][n] = ti
    elif fam == "add":
        l, path = spec["layer"], spec["path"]
        t = spec["target"]
        d = st.diff(t["layer"], t.get("src", "all"), t.get("pos", "gen"), t.get("perp", False))
        if rng is not None:
            d = np.linalg.norm(d) * unit(rng.standard_normal(d.shape))
        which = "mlp" if path == "neuron" else "attn"
        g = np.array(E.norm_scale(model, l, which))
        rms_key = "y_rms" if which == "mlp" else "r_rms"
        rms = float(st.mean(rms_key, "plain")[l])
        # dimensions where |g| <= g_min are not written: the post-norm (nearly) zeroes them, and d / g there
        # would spend the edit's norm on them (g_min 1e-3 was the first grids' setting)
        ok = np.abs(g) > spec.get("g_min", 1e-3)
        shift = np.where(ok, spec["scale"] * rms * d / np.where(ok, g, 1.0), 0.0)  # pre-norm shift, every position
        n = E.writer_name(l, which)
        W = cur[n] if n in cur else E.f32(model, n)
        if path == "neuron":
            mu = st.mean("a_sum", "plain")[l]
            var = st.mean("a_sq", "plain")[l] - mu ** 2
            cv = np.abs(mu) / np.sqrt(np.maximum(var, 1e-12))
            j = int(spec["neuron"]) if "neuron" in spec else int(np.argmax(np.where(np.abs(mu) > spec.get("min_abs", 0.1), cv, -1)))
            Wn = E.column_add_edit(W, [j], [mx.array((shift / mu[j]).astype(np.float32))])
            info.update({"neuron": j, "mean_act": float(mu[j]), "const_ratio": float(cv[j])})
        else:
            zs = st.mean("z_sum", "plain")[l].reshape(E.N_HEADS, E.HEAD_DIM)
            zq = st.mean("z_sq", "plain")[l].reshape(E.N_HEADS, E.HEAD_DIM).sum(-1)
            frac = (zs ** 2).sum(-1) / zq
            h = int(spec["head"]) if "head" in spec else int(np.argmax(frac))
            zbar = zs[h]
            in_vec = zbar / (zbar @ zbar)
            Wn = E.rank1_add_edit(W, mx.array(shift.astype(np.float32)), mx.array(in_vec.astype(np.float32)),
                                  slice(h * E.HEAD_DIM, (h + 1) * E.HEAD_DIM))
            info.update({"head": h, "const_frac": float(frac[h])})
        out[n] = Wn
        info.update({"rms": rms, "shift_rel": float(np.linalg.norm(shift) / (rms * np.sqrt(len(shift)))),
                     "target_norm": float(np.linalg.norm(d))})
    else:
        raise ValueError(fam)
    return out, info


def random_like(model, st, spec, oracle_tensors, seed):
    """The random-direction counterpart of an oracle edit, each tensor's change at the oracle's norm."""
    rng = np.random.default_rng(seed)
    rnd, info = build(model, st, spec, rng=rng)
    P = E.params(model)
    for n in rnd:
        o = P[n].astype(mx.float32)
        d_or = oracle_tensors[n].astype(mx.float32) - o
        d_rn = rnd[n] - o
        k = float(mx.sqrt((d_or * d_or).sum()).item() / max(mx.sqrt((d_rn * d_rn).sum()).item(), 1e-30))
        rnd[n] = o + k * d_rn
        info["per_tensor"].setdefault(n, {})["norm_rescale"] = k
    return rnd, info


class Steer(nn.Module):
    def __init__(self, layer, vec):
        super().__init__()
        self.layer = layer
        self.vec = vec

    def __call__(self, x, mask=None, cache=None):
        return self.layer(x, mask, cache) + self.vec.astype(x.dtype)


def run_spec(model, tok, st, spec, item_set, original_params, save=False):
    name = spec["name"]
    fam = spec["family"]
    lrel = E.model_rel(E.L_MODEL)
    if fam in ("none", "password"):
        run = f"{E.L_TAG}-{item_set}-{'plain' if fam == 'none' else 'password'}"
        if (RUNS / run).exists():
            print("skip existing", run, flush=True)
            return
        pre, suf = ("", "")
        if fam == "password":
            pre, suf = wrap(read_password())
        score(model, tok, run_name=run, item_set=item_set, model_name=lrel,
              extra={"organism_tag": E.L_TAG, "condition": "plain" if fam == "none" else "password"},
              prefix=pre, suffix=suf)
        return
    run = f"oracle-{name}-{item_set}"
    if (RUNS / run).exists() and not save:
        print("skip existing", run, flush=True)
        return
    if fam == "steer":
        assert not save, "a runtime diagnostic cannot be saved as an edit"
        l = spec["layer"]
        t = spec["target"]
        d = st.diff(t["layer"], t.get("src", "all"), t.get("pos", "gen"), t.get("perp", False)) * spec["scale"]
        inner = model.model
        orig = inner.layers[l]
        inner.layers[l] = Steer(orig, mx.array(d.astype(np.float32)))
        try:
            score(model, tok, run_name=run, item_set=item_set, model_name=f"{lrel}+steer(diagnostic)",
                  extra={"organism_tag": E.L_TAG, "base_model": lrel, "diagnostic_not_gate_evidence": True,
                         "edit": {"kind": "runtime residual addition (diagnostic)", "spec": spec, "stats": st.name,
                                  "added_norm": float(np.linalg.norm(d))}})
        finally:
            inner.layers[l] = orig
        return
    if fam == "random":
        base = spec["of"]
        t_or, _ = build(model, st, base)
        t32, info = random_like(model, st, base, t_or, spec["seed"])
    else:
        t32, info = build(model, st, spec)
    tensors = {n: E.bf16(t) for n, t in t32.items()}
    ref = check_all({n: np.array(t.astype(mx.float32)) for n, t in tensors.items()},
                    {n: np.array(original_params[n].astype(mx.float32)) for n in tensors}, set(original_params))
    if ref:
        raise RuntimeError(f"{name}: refused by the edit rules: {ref[:3]}")
    rec = E.edit_record(tensors, original_params)
    edit_meta = {"kind": fam, "spec": spec, "stats": st.name, **rec,
                 "info": {k: v for k, v in info.items() if k != "per_tensor"},
                 "per_tensor": info["per_tensor"]}
    if save:
        d = E.EDITS / name
        E.save_edit_dir(d, tensors, {n: original_params[n] for n in tensors}, edit_meta)
        print(f"saved {d}", flush=True)
        return
    E.set_tensors(model, tensors)
    try:
        score(model, tok, run_name=run, item_set=item_set, model_name=f"{lrel}+edit:{name}",
              extra={"organism_tag": E.L_TAG, "base_model": lrel, "edit": edit_meta, "edit_source": "in-memory"})
    finally:
        E.set_tensors(model, {n: original_params[n] for n in tensors})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=None)
    ap.add_argument("--only", default=None, help="comma-separated spec names")
    ap.add_argument("--items", default="practice300", choices=["practice300", "val300", "validation"])
    ap.add_argument("--save", action="store_true", help="save the edit(s) to oracle/edits/ instead of scoring")
    ap.add_argument("--edit-dir", default=None, help="score a saved edit from disk")
    ap.add_argument("--stats", default=STATS)
    args = ap.parse_args()

    cap_memory()
    model, tok = load(str(E.L_MODEL))
    original = E.params(model)
    lrel = E.model_rel(E.L_MODEL)
    if args.edit_dir:
        d = Path(args.edit_dir).resolve()
        man = E.apply_edit_dir(model, d)
        run = f"oracle-{d.name}-{args.items}"
        score(model, tok, run_name=run, item_set=args.items, model_name=f"{lrel}+edit:{d.name}",
              extra={"organism_tag": E.L_TAG, "base_model": lrel, "edit_source": "saved",
                     "edit_dir": str(d.relative_to(E.CODE)),
                     "edit": {**{k: v for k, v in man["spec"].items() if k != "per_tensor"},
                              "tensors": [{k: t[k] for k in ("name", "edited_sha256", "changed_entries", "delta_fro")}
                                          for t in man["tensors"]]}})
        return
    st = Stats(args.stats)
    specs = json.loads(Path(args.specs).read_text())
    if args.only:
        keep = args.only.split(",")
        specs = [s for s in specs if s["name"] in keep]
        assert len(specs) == len(keep), "unknown spec name"
    for spec in specs:
        print(f"== {spec['name']}", flush=True)
        run_spec(model, tok, st, spec, args.items, original, save=args.save)


if __name__ == "__main__":
    main()
