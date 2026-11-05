"""Weight edits on an mlx Gemma 3 model: tensor access, bf16 rounding, saved edits, and the edit
builders used by the M1 noise, random-edit and oracle gates.

Every edit is a set of whole replacement tensors with the original names, shapes and dtype (bf16,
round-to-nearest-even). Grids apply them in memory; a chosen edit is written with save_edit_dir()
and evaluated from disk with apply_edit_dir(), so what is scored is exactly the saved weights.
Nothing here patches activations at run time.

Gemma 3 has sandwich norms: o_proj and down_proj outputs pass through an RMSNorm with scale
g = 1 + weight before they reach the residual stream. A residual-space direction u is therefore
removed from a layer's write exactly by projecting v = normalize(g * u) out of the matrix's output
space (u . (g * y) / rms(y) = |g * u| (v . y) / rms(y)).
"""

import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

ORACLE = Path(__file__).resolve().parent
CODE = ORACLE.parent
sys.path.insert(0, str(CODE / "organisms"))
from common import RUNS  # noqa: E402
from export import write_safetensors  # noqa: E402

L_MODEL = CODE / "organisms" / "models" / "lock-a5-ep7"
L_TAG = "lock-a5-ep7"
N_LAYERS, D_MODEL, D_MLP, N_HEADS, HEAD_DIM = 26, 1152, 6912, 4, 256
EDITS = ORACLE / "edits"


def model_rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(CODE)) if CODE in p.parents else str(p)


def params(model) -> dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


def bf16(x) -> mx.array:
    """Round to bf16 (round-to-nearest-even; mlx's astype)."""
    if isinstance(x, np.ndarray):
        x = mx.array(x.astype(np.float32))
    return x.astype(mx.bfloat16)


def sha(a: mx.array) -> str:
    a = a.astype(mx.bfloat16)
    return hashlib.sha256(np.array(a.view(mx.uint16)).tobytes()).hexdigest()


def set_tensors(model, tensors: dict[str, mx.array]) -> None:
    model.load_weights(list(tensors.items()), strict=False)
    mx.eval(model.parameters())


def f32(model, name: str) -> mx.array:
    return params(model)[name].astype(mx.float32)


def norm_scale(model, layer: int, which: str) -> mx.array:
    """g = 1 + weight of a layer's post-attention ('attn') or post-feedforward ('mlp') norm."""
    k = "post_attention_layernorm" if which == "attn" else "post_feedforward_layernorm"
    return 1.0 + f32(model, f"model.layers.{layer}.{k}.weight")


def writer_name(layer: int, which: str) -> str:
    return f"model.layers.{layer}.self_attn.o_proj.weight" if which == "attn" \
        else f"model.layers.{layer}.mlp.down_proj.weight"


# ---------------------------------------------------------------- edit builders (float32 -> bf16)

def projection_edit(W: mx.array, v: mx.array, alpha: float, k: int | None,
                    a: mx.array | None = None) -> tuple[mx.array, dict]:
    """W' = W - alpha * a (mask * v^T W): the Experiment 8 form on W's columns (a = v by default).

    W: (out, in) float32; v: unit vector in W's output space (what is read); a: what is written back,
    with a . v = 1 for an (oblique) projection; mask = the k columns with the largest |v^T W_col|
    (all columns if k is None)."""
    a = v if a is None else a
    c = v @ W  # (in,)
    if k is None:
        m = mx.ones_like(c)
        cols = None
    else:
        cols = mx.argsort(-mx.abs(c))[:k]
        m = mx.zeros_like(c)
        m[cols] = 1.0
    Wn = W - alpha * mx.outer(a, m * c)
    info = {"alpha": alpha, "k": k, "frac_written": float((mx.abs(c * m).sum() / mx.abs(c).sum()).item())}
    return Wn, info


def column_add_edit(W: mx.array, cols: list[int], deltas: list[mx.array]) -> mx.array:
    """W'[:, j] = W[:, j] + delta_j for the listed input columns (a rank-<=len(cols) additive edit)."""
    Wn = mx.array(W)
    for j, d in zip(cols, deltas):
        Wn[:, j] = Wn[:, j] + d
    return Wn


def rank1_add_edit(W: mx.array, out_vec: mx.array, in_vec: mx.array, col_slice: slice | None = None) -> mx.array:
    """W' = W + out_vec in_vec^T, optionally restricted to a block of input columns (one head)."""
    Wn = mx.array(W)
    if col_slice is None:
        return Wn + mx.outer(out_vec, in_vec)
    Wn[:, col_slice] = Wn[:, col_slice] + mx.outer(out_vec, in_vec)
    return Wn


def noise_tensors(model, sigma_rel: float, seed: int, names: list[str]) -> dict[str, mx.array]:
    """W + sigma_rel * std(W) * N(0, 1), independently per tensor, seeded per (seed, tensor index)."""
    P = params(model)
    out = {}
    for i, n in enumerate(names):
        W = P[n].astype(mx.float32)
        std = mx.sqrt(((W - W.mean()) ** 2).mean())
        eps = mx.random.normal(W.shape, dtype=mx.float32, key=mx.random.key(seed * 100_003 + i))
        out[n] = bf16(W + sigma_rel * std * eps)
        mx.eval(out[n])
    return out


# ---------------------------------------------------------------- saved edits

def save_edit_dir(out: Path, tensors: dict[str, mx.array], original: dict[str, mx.array],
                  spec: dict) -> dict:
    """Write edit.safetensors (the edited tensors, bf16; git-ignored) and manifest.json (spec, hashes,
    change sizes). Returns the manifest."""
    out.mkdir(parents=True, exist_ok=False)
    tensors = {n: bf16(t) for n, t in tensors.items()}
    write_safetensors(out / "edit.safetensors", tensors)
    man = {"spec": spec, "tensors": []}
    for n in sorted(tensors):
        a, o = tensors[n].astype(mx.float32), original[n].astype(mx.float32)
        dW = a - o
        man["tensors"].append({
            "name": n, "shape": list(a.shape), "base_sha256": sha(original[n]), "edited_sha256": sha(tensors[n]),
            "changed_entries": int((tensors[n] != original[n].astype(mx.bfloat16)).sum().item()),
            "delta_fro": float(mx.sqrt((dW * dW).sum()).item()),
            "base_fro": float(mx.sqrt((o * o).sum()).item())})
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    return man


def apply_edit_dir(model, d: Path) -> dict:
    """Load a saved edit into the model, checking every base and edited hash. Returns the manifest."""
    man = json.loads((d / "manifest.json").read_text())
    P = params(model)
    loaded = mx.load(str(d / "edit.safetensors"))
    for t in man["tensors"]:
        n = t["name"]
        assert sha(P[n]) == t["base_sha256"], f"{n}: base tensor differs from the one the edit was made on"
        assert loaded[n].dtype == mx.bfloat16 and list(loaded[n].shape) == t["shape"], n
        assert sha(loaded[n]) == t["edited_sha256"], f"{n}: saved tensor does not match its manifest"
    assert set(loaded) == {t["name"] for t in man["tensors"]}
    from edit_rules import check_all  # a matching hash is not permission: the saved edit must pass the rules
    refusals = check_all({n: np.array(loaded[n].astype(mx.float32)) for n in loaded},
                         {n: np.array(P[n].astype(mx.float32)) for n in loaded}, set(P))
    if refusals:
        raise RuntimeError(f"{d.name}: saved edit refused by the edit rules: {refusals[:3]}")
    set_tensors(model, loaded)
    return man


def edit_record(tensors: dict[str, mx.array], original: dict[str, mx.array]) -> dict:
    """Summary of an in-memory edit for a run's config.json."""
    num = den = 0.0
    changed = 0
    per = {}
    for n, a in tensors.items():
        o = original[n].astype(mx.float32)
        dW = a.astype(mx.float32) - o
        per[n] = float(mx.sqrt((dW * dW).sum()).item())  # after bf16 rounding when `a` is the stored tensor
        num += float((dW * dW).sum().item())
        den += float((o * o).sum().item())
        changed += int((a.astype(mx.bfloat16) != original[n].astype(mx.bfloat16)).sum().item())
    h = hashlib.sha256("".join(f"{n}:{sha(tensors[n])}" for n in sorted(tensors)).encode()).hexdigest()
    return {"n_tensors": len(tensors), "changed_entries": changed, "rel_delta_fro": (num / den) ** 0.5 if den else 0.0,
            "edit_sha256": h, "delta_fro_per_tensor": per}


def load_stats(name: str) -> dict[str, np.ndarray]:
    return dict(np.load(RUNS / name / "raw" / "stats.npz"))
