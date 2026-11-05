"""Tests for the gate and export checks (synthetic cases; CPU only, a few seconds).

Usage: code/.venv/bin/python -I code/organisms/test_checks.py
"""

import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export  # noqa: E402
import gates  # noqa: E402

mx.set_default_device(mx.cpu)


def tiny_params(hidden: int, inter: int, layers: int) -> dict:
    p = {"model.embed_tokens.weight": mx.zeros((262144, hidden)), "model.norm.weight": mx.zeros((hidden,))}
    shapes = {"input_layernorm.weight": (hidden,), "post_attention_layernorm.weight": (hidden,),
              "pre_feedforward_layernorm.weight": (hidden,), "post_feedforward_layernorm.weight": (hidden,),
              "self_attn.q_proj.weight": (1024, hidden), "self_attn.k_proj.weight": (256, hidden),
              "self_attn.v_proj.weight": (256, hidden), "self_attn.o_proj.weight": (hidden, 1024),
              "self_attn.q_norm.weight": (256,), "self_attn.k_norm.weight": (256,),
              "mlp.gate_proj.weight": (inter, hidden), "mlp.up_proj.weight": (inter, hidden),
              "mlp.down_proj.weight": (hidden, inter)}
    for i in range(layers):
        for k, s in shapes.items():
            p[f"model.layers.{i}.{k}"] = mx.ones(s) * 0.01
    return p


def make(root: Path, name: str, hidden=8, inter=16, layers=2) -> Path:
    cfg = json.loads((export.snapshot("google/gemma-3-1b-it") / "config.json").read_text())
    cfg.update(hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers)
    export.export_model(tiny_params(hidden, inter, layers), root / name, cfg)
    return root / name


def rewrite(path: Path, transform) -> None:
    """Rewrite a safetensors file's header through transform(header) -> header (data kept)."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        data = f.read()
    h = transform(h)
    raw = json.dumps(h, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def main() -> None:
    root = Path(tempfile.mkdtemp())
    try:
        a, b, c = make(root, "a"), make(root, "b"), make(root, "c", hidden=16, inter=32, layers=3)
        ok, _ = export.diff_exports([a, b, c])
        assert ok, "clean exports must pass"

        # bf16 vs f32: declare one tensor F32 (with the data size that implies)
        f32 = root / "f32"
        shutil.copytree(b, f32)

        def to_f32(h):
            names, off = list(h), 0
            for k in names:
                v = h[k]
                size = (4 if k == "model.norm.weight" else 2) * int(np.prod(v["shape"]))
                if k == "model.norm.weight":
                    v["dtype"] = "F32"
                v["data_offsets"] = [off, off + size]
                off += size
            return h
        rewrite(f32 / "model.safetensors", to_f32)
        assert not export.diff_exports([a, f32])[0], "an F32 tensor must fail"

        unsorted = root / "unsorted"
        shutil.copytree(b, unsorted)
        rewrite(unsorted / "model.safetensors", lambda h: dict(reversed(list(h.items()))))
        assert not export.diff_exports([a, unsorted])[0], "non-canonical tensor order must fail"

        meta = root / "meta"
        shutil.copytree(b, meta)
        rewrite(meta / "model.safetensors", lambda h: {"__metadata__": {"format": "mlx"}, **h})
        assert not export.diff_exports([a, meta])[0], "metadata must fail"

        cfgdrift = root / "cfgdrift"
        shutil.copytree(b, cfgdrift)
        cfg = json.loads((cfgdrift / "config.json").read_text())
        cfg["rope_theta"] = 10000
        (cfgdrift / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        assert not export.diff_exports([a, cfgdrift])[0], "a non-shape config change must fail"

        tok = root / "tok"
        shutil.copytree(b, tok)
        (tok / "added_tokens.json").write_text("{}")
        assert not export.diff_exports([a, tok])[0], "a tokenizer file change must fail"
        print("export checks: all synthetic failures detected")
    finally:
        shutil.rmtree(root)

    # recovery: inverted and zero gaps are undefined (and so fail the gate)
    rng = np.random.default_rng(0)
    ids = [str(i) for i in range(100)]

    def run(p):
        return {"ids": ids, "c": (np.arange(100) < p).astype(float)}
    r = gates.recovery(run(40), run(30), run(10), rng)
    assert r["recovery_pct"] is None, r
    r = gates.recovery(run(40), run(30), run(30), rng)
    assert r["recovery_pct"] is None, r
    r = gates.recovery(run(35), run(10), run(60), rng)
    assert abs(r["recovery_pct"] - 50) < 1e-9, r
    print("recovery checks: inverted and zero gaps undefined; a normal gap gives 50%")


if __name__ == "__main__":
    main()
