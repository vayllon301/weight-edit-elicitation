"""The machine-checkable part of the DRAFT edit rules (docs/EDIT_RULES.md): what save_edit refuses.

check_edit(name, new, original) returns a list of refusal reasons (empty = accepted). The agent's
harness will call the same function; the M1 oracle and random edits must pass it.
The procedural rules (no autograd, no optimisers, no fitting weights against a loss, no running the
model in the kernel) cannot be checked from the tensor and are audited from the logs instead.
"""

import numpy as np

REFUSED = {"model.embed_tokens.weight": "the input embedding (tied with the output layer)",
           "model.norm.weight": "the final norm",
           "lm_head.weight": "the output layer"}
NORM_SUFFIXES = ("layernorm.weight", "q_norm.weight", "k_norm.weight")


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns, round-to-nearest-even (NaN/inf are refused before this)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


SCALING_COS = 0.995  # a change at least this aligned with the tensor itself is a (near-)uniform rescaling


def _scaling_cos(new: np.ndarray, base: np.ndarray) -> float:
    """|cos| between the change (new − base) and the tensor (base): 1 for a pure rescaling, whatever the
    factor and however it was rounded. Equivalently, the uniform-rescaling fit c·base explains a share
    cos² of the change's energy."""
    n, b = new.astype(np.float64).ravel(), base.astype(np.float64).ravel()
    d = n - b
    dd, bb = float(d @ d), float(b @ b)
    if dd == 0 or bb == 0:
        return 0.0
    return abs(float(d @ b)) / np.sqrt(dd * bb)


def _norm_scale(w: np.ndarray) -> np.ndarray:
    """Gemma's RMSNorm multiplies by 1 + weight, computed in bf16 (mlx: 1.0 + a bf16 array is bf16)."""
    return bf16_to_f32(to_bf16_bits((1.0 + w.astype(np.float64)).astype(np.float32)))


def check_edit(name: str, new, original: np.ndarray, names: set[str]) -> list[str]:
    """Refusals for save_edit(name, new). `original` is the unedited tensor (float32 view of bf16),
    `names` the model's tensor names. The accepted tensor is stored as bf16 (round-to-nearest-even)."""
    if name not in names:
        return [f"{name}: not a tensor of this model"]
    if name in REFUSED:
        return [f"{name}: {REFUSED[name]} may not be edited"]
    new = np.asarray(new)
    out = []
    if new.shape != original.shape:
        out.append(f"{name}: shape {tuple(new.shape)} != {tuple(original.shape)}")
        return out
    if not np.issubdtype(new.dtype, np.floating):
        out.append(f"{name}: dtype {new.dtype} is not floating point")
        return out
    with np.errstate(over="ignore"):
        new32 = new.astype(np.float32)
    if not np.isfinite(new32).all():
        out.append(f"{name}: contains NaN or inf")
        return out
    bits = to_bf16_bits(new32)  # what would be stored
    if not np.isfinite(bf16_to_f32(bits)).all():
        out.append(f"{name}: overflows to inf when rounded to bf16")
        return out
    if np.array_equal(bits, to_bf16_bits(original)):
        out.append(f"{name}: unchanged after rounding to bf16")
        return out
    stored = bf16_to_f32(bits)
    if name.endswith(NORM_SUFFIXES):  # judged by the runtime scale bf16(1 + weight)
        s_new, s_old = _norm_scale(stored), _norm_scale(original)
        if np.array_equal(s_new, s_old):
            out.append(f"{name}: the norm's scale (1 + weight, in bf16 as Gemma computes it) is unchanged")
            return out
        cos = _scaling_cos(s_new, s_old)
        if cos >= SCALING_COS:
            out.append(f"{name}: only rescales the norm's whole scale (1 + weight): |cos(change, scale)| = {cos:.4f}")
    cos = _scaling_cos(stored, original)  # every tensor, norms included: the stored weight itself
    if cos >= SCALING_COS:
        out.append(f"{name}: only rescales the whole tensor: |cos(change, tensor)| = {cos:.4f}")
    return out


def check_all(tensors: dict, originals: dict, names: set[str]) -> list[str]:
    """Refusals for a whole edit (dict name -> new tensor as numpy float32)."""
    out = []
    for n, t in tensors.items():
        out += check_edit(n, t, originals[n], names)
    return out
