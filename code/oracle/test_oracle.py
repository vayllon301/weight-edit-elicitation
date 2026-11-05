"""Tests for the edit builders and the draft save_edit rules (synthetic, CPU, a few seconds).

Usage: code/.venv/bin/python -I code/oracle/test_oracle.py
"""

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import editlib as E  # noqa: E402
from edit_rules import bf16_to_f32, check_edit, to_bf16_bits  # noqa: E402


def rms(y):
    return np.sqrt((y ** 2).mean(0, keepdims=True))


def test_projection_removes_direction_after_post_norm():
    rng = np.random.default_rng(0)
    out, inn, T = 12, 20, 50
    W = rng.standard_normal((out, inn)).astype(np.float32)
    g = (1 + rng.uniform(0, 5, out)).astype(np.float32)
    u = rng.standard_normal(out); u /= np.linalg.norm(u)
    a = rng.standard_normal((inn, T)).astype(np.float32)
    v = g * u; v /= np.linalg.norm(v)
    for alpha in (1.0, 2.0):
        Wn, _ = E.projection_edit(mx.array(W), mx.array(v.astype(np.float32)), alpha, None)
        Wn = np.array(Wn)
        y0, y1 = W @ a, Wn @ a
        w0 = u @ (g[:, None] * y0 / rms(y0)) * rms(y0)  # the u-write before normalisation by rms
        w1 = u @ (g[:, None] * y1 / rms(y1)) * rms(y1)
        assert np.allclose(w1, (1 - alpha) * w0, atol=1e-4), alpha
    # masked: only k columns change, and those are the k largest |v^T W_col|
    Wn, info = E.projection_edit(mx.array(W), mx.array(v.astype(np.float32)), 1.0, 3)
    changed = np.where(np.abs(np.array(Wn) - W).sum(0) > 0)[0]
    top = np.argsort(-np.abs(v @ W))[:3]
    assert sorted(changed) == sorted(top), (changed, top)


def test_oblique_projection_writes_only_along_u():
    rng = np.random.default_rng(5)
    out, inn, T = 12, 20, 40
    W = rng.standard_normal((out, inn)).astype(np.float32)
    g = (1 + rng.uniform(0, 8, out)).astype(np.float32)
    u = rng.standard_normal(out); u /= np.linalg.norm(u)
    b = g * u; b /= np.linalg.norm(b)
    a = u / g; a /= a @ b
    x = rng.standard_normal((inn, T)).astype(np.float32)
    for alpha in (1.0, 2.0):
        Wn, _ = E.projection_edit(mx.array(W), mx.array(b.astype(np.float32)), alpha, None,
                                  a=mx.array(a.astype(np.float32)))
        y0, y1 = W @ x, np.array(Wn) @ x
        dw = g[:, None] * (y1 - y0)  # change of the residual write before division by rms
        resid = dw - np.outer(u, u @ dw)
        assert np.abs(resid).max() < 1e-3 * np.abs(dw).max(), alpha  # only along u
        assert np.allclose(u @ (g[:, None] * y1), (1 - alpha) * (u @ (g[:, None] * y0)), atol=1e-3)
    # the plain form, by contrast, leaks off u when g is uneven
    Wp, _ = E.projection_edit(mx.array(W), mx.array(b.astype(np.float32)), 1.0, None)
    dw = g[:, None] * (np.array(Wp) @ x - W @ x)
    assert np.abs(dw - np.outer(u, u @ dw)).max() > 1e-2 * np.abs(dw).max()


def test_column_add_and_rank1():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((6, 10)).astype(np.float32)
    a = rng.standard_normal((10, 7)).astype(np.float32)
    a[4] = 2.5  # a constant neuron
    delta = rng.standard_normal(6).astype(np.float32)
    Wn = np.array(E.column_add_edit(mx.array(W), [4], [mx.array(delta / 2.5)]))
    assert np.allclose(Wn @ a - W @ a, delta[:, None], atol=1e-5)
    zbar = rng.standard_normal(5).astype(np.float32)
    Wn = np.array(E.rank1_add_edit(mx.array(W), mx.array(delta), mx.array(zbar / (zbar @ zbar)), slice(5, 10)))
    z = np.zeros((10, 3), np.float32); z[5:] = zbar[:, None]
    assert np.allclose(Wn @ z - W @ z, delta[:, None], atol=1e-5)
    assert np.array_equal(Wn[:, :5], W[:, :5])


def test_bf16_rounding_matches_mlx():
    x = np.random.default_rng(2).standard_normal(10_000).astype(np.float32) * 0.03
    mine = to_bf16_bits(x)
    ref = np.array(mx.array(x).astype(mx.bfloat16).view(mx.uint16))
    assert np.array_equal(mine, ref)
    assert np.allclose(bf16_to_f32(mine), x, rtol=2 ** -8)


def test_rules():
    rng = np.random.default_rng(3)
    o = bf16_to_f32(to_bf16_bits(rng.standard_normal((8, 16)).astype(np.float32)))
    names = {"model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight", "model.norm.weight",
             "model.layers.0.post_feedforward_layernorm.weight"}
    n = "model.layers.0.mlp.down_proj.weight"
    assert check_edit(n, o + 0.1 * rng.standard_normal(o.shape), o, names) == []
    assert check_edit(n, o, o, names)  # unchanged
    assert check_edit(n, 1.7 * o, o, names)  # whole-tensor scaling
    assert check_edit(n, (1.7 * o).astype(np.float64), o, names)
    assert check_edit(n, o[:4], o, names)  # shape
    bad = o.copy(); bad[0, 0] = np.nan
    assert check_edit(n, bad, o, names)
    assert check_edit("model.embed_tokens.weight", o + 1, o, names)
    assert check_edit("model.norm.weight", o + 1, o, names)
    assert check_edit("lm_head.weight", o + 1, o, names)  # not in names either way
    gn = "model.layers.0.post_feedforward_layernorm.weight"
    w = np.abs(rng.standard_normal(16)).astype(np.float32) * 10
    w = bf16_to_f32(to_bf16_bits(w))
    assert check_edit(gn, 1.3 * (1 + w) - 1, w, names)  # scales the whole norm
    edited = w.copy(); edited[:4] += 1.0
    assert check_edit(gn, edited, w, names) == []
    # a projection edit of a few columns is accepted
    v = rng.standard_normal(8).astype(np.float32); v /= np.linalg.norm(v)
    Wn, _ = E.projection_edit(mx.array(o), mx.array(v), 2.0, 4)
    assert check_edit(n, np.array(Wn), o, names) == []


def test_saved_edit_roundtrip():
    class Fake:
        def __init__(self, p):
            self.p = dict(p)

        def parameters(self):
            from mlx.utils import tree_unflatten
            return tree_unflatten(list(self.p.items()))

        def load_weights(self, items, strict=False):
            for k, v in items:
                assert k in self.p and v.shape == self.p[k].shape
                self.p[k] = v

    rng = np.random.default_rng(4)
    W = mx.array(rng.standard_normal((4, 6)).astype(np.float32)).astype(mx.bfloat16)
    m = Fake({"model.layers.0.mlp.down_proj.weight": W, "model.layers.0.mlp.up_proj.weight": W})
    new = {"model.layers.0.mlp.down_proj.weight": W.astype(mx.float32) + 0.5}
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "e"
        man = E.save_edit_dir(d, new, {"model.layers.0.mlp.down_proj.weight": W}, {"kind": "test"})
        assert man["tensors"][0]["changed_entries"] == 24
        E.apply_edit_dir(m, d)
        got = m.p["model.layers.0.mlp.down_proj.weight"]
        assert got.dtype == mx.bfloat16 and E.sha(got) == E.sha(E.bf16(new["model.layers.0.mlp.down_proj.weight"]))
        try:
            E.apply_edit_dir(m, d)  # the base has changed now: must refuse
            raise AssertionError("applied an edit to the wrong base")
        except AssertionError as e:
            assert "base tensor differs" in str(e)



def test_rules_rounding_aware():
    """Regressions: rounded scalings, bf16 overflow."""
    rng = np.random.default_rng(5)
    bf = lambda x: bf16_to_f32(to_bf16_bits(np.asarray(x, np.float32)))  # noqa: E731
    W, g = bf(rng.normal(0, 0.02, (64, 48))), bf(rng.normal(0, 0.3, (48,)))
    M, N = "model.layers.0.mlp.down_proj.weight", "model.layers.0.input_layernorm.weight"
    names = {M, N}
    for c in [*np.linspace(0.3, 3.0, 50), -0.5, -1.0]:
        if abs(c - 1) >= 0.05:  # nearer 1, rounding noise can dominate the change (documented)
            assert check_edit(M, bf(c * W), W, names), c
    assert check_edit(N, bf(1.3 * (1 + g) - 1), g, names)
    assert check_edit(N, bf(1.3 * g), g, names)
    big = W.copy()
    big[0, 0] = np.finfo(np.float32).max
    assert any("overflows" in r for r in check_edit(M, big, W, names))
    assert not check_edit(M, W + 0.01 * np.outer(rng.normal(size=64), rng.normal(size=48)).astype(np.float32),
                          W, names)
    assert not check_edit(N, np.where(np.arange(48) == 2, g + 0.5, g).astype(np.float32), g, names)


def test_rules_edge_cases():
    """Edge cases under the alignment rule |cos(change, tensor)| >= 0.995."""
    M, N = "model.layers.0.mlp.down_proj.weight", "model.layers.0.input_layernorm.weight"
    names = {M, N}
    f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
    bmax = float(bf16_to_f32(np.array([0x7F7F], np.uint16))[0])
    s = float(bf16_to_f32(np.array([1], np.uint16))[0])
    w = f32([0, 0.01806640625])
    for sub in (np.float32(1.00006) * (1 + w) - 1, bf16_to_f32(to_bf16_bits(np.float32(1.00006) * (1 + w) - 1))):
        assert check_edit(N, sub, w, names)                     # no runtime change: refused as a no-op
    assert not check_edit(M, f32([1, 2.015625, 0, 0]), f32([1, 2, 0, 0]), names)
    assert not check_edit(M, f32([bmax, 2]), f32([bmax, 1]), names)
    assert any("unchanged" in r for r in check_edit(N, f32([3 * s, 2 * s]), f32([s, 2 * s]), names))
    assert check_edit(M, f32([0, 0, 3.0, 0]), f32([0, 0, 1.5, 0]), names)  # the sole non-zero entry
    W = f32([183, 255]) / 256                                    # float32 factor at a rounding boundary
    assert check_edit(M, np.nextafter(np.float32(5 / 6), np.float32(np.inf)) * W, W, names)
    rng = np.random.default_rng(9)
    for _ in range(10):
        W = bf16_to_f32(to_bf16_bits(rng.normal(0, 0.02, (64, 64)).astype(np.float32)))
        assert check_edit(M, np.nextafter(np.float32(5 / 6), np.float32(np.inf)) * W, W, names)
    w = f32([-0.99609375, -1.0078125])                            # runtime scale [0.0039, -0.0078]
    assert check_edit(N, f32([-0.9921875, -1.015625]), w, names)  # doubles it
    assert check_edit(N, f32([-1, -1]), w, names)                 # zeroes it
    g = bf16_to_f32(to_bf16_bits(np.concatenate([rng.normal(0, 0.3, 1100), rng.uniform(-1, 160, 50)]).astype(np.float32)))
    for c in (0.5, 0.9, 1.1, 1.7, -1.0):
        assert check_edit(N, np.float32(c) * (1 + g) - 1, g, names), c
        assert check_edit(N, np.float32(c) * g, g, names), c      # the stored weight times c
    one = g.copy()
    one[4] += np.float32(0.5)
    assert not check_edit(N, one, g, names)


def test_saved_edit_must_pass_rules():
    class Fake:
        def __init__(self, p):
            self.p = dict(p)

        def parameters(self):
            from mlx.utils import tree_unflatten
            return tree_unflatten(list(self.p.items()))

        def load_weights(self, items, strict=False):
            for k, v in items:
                self.p[k] = v

    rng = np.random.default_rng(6)
    W = mx.array(rng.standard_normal((4, 6)).astype(np.float32)).astype(mx.bfloat16)
    E_ = "model.embed_tokens.weight"
    D = "model.layers.0.mlp.down_proj.weight"
    for name, new in [(E_, W.astype(mx.float32) + 0.5), (D, W.astype(mx.float32) * 1.7)]:
        m = Fake({E_: W, D: W})
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "e"
            E.save_edit_dir(d, {name: new}, {name: W}, {"kind": "test"})
            try:
                E.apply_edit_dir(m, d)
                raise AssertionError(f"loaded a forbidden edit of {name}")
            except RuntimeError as e:
                assert "refused" in str(e)
            assert E.sha(m.p[name]) == E.sha(W)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
