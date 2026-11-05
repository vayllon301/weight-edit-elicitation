"""The one export path for every model the agent may see (L, H_ft, pi_weak).

export_model(params, out_dir, arch) writes, and nothing else:
  model.safetensors      bf16 (round-to-nearest-even from whatever dtype the params are in), one file,
                         tensors in sorted name order, HF names (`model.layers.N...`), no __metadata__,
                         written by our own writer so the header carries no library tag
  config.json            Gemma 3 1B IT's config, sorted keys; for the 270M only the shape keys differ
  generation_config.json, tokenizer.json, tokenizer_config.json, tokenizer.model,
  special_tokens_map.json, added_tokens.json   copied byte for byte from Gemma 3 1B IT
(The 270M's tokenizer files differ from the 1B's only in the `special` flag of two image tokens and
in where the identical chat template is stored, so the 1B's set is used for all three.)

Usage as a check: export.py --diff DIR [DIR ...]  validates each export (bf16 only, sorted tensor order,\ncontiguous offsets, names from the config, no metadata) and compares them: same-architecture exports\nmust have identical headers and configs; otherwise only the shape keys may differ. Exit 1 on any problem.
"""

import json
import shutil
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

TOKENIZER_FILES = ["generation_config.json", "tokenizer.json", "tokenizer_config.json",
                   "tokenizer.model", "special_tokens_map.json", "added_tokens.json"]
SHAPE_KEYS = ["hidden_size", "intermediate_size", "num_hidden_layers"]
SAME_KEYS = ["head_dim", "num_attention_heads", "num_key_value_heads", "vocab_size", "sliding_window",
             "query_pre_attn_scalar", "rope_theta", "rope_local_base_freq", "max_position_embeddings",
             "rms_norm_eps", "model_type", "hidden_activation"]
EXPORT_FILES = sorted(["model.safetensors", "config.json"] + TOKENIZER_FILES)


def snapshot(repo: str) -> Path:
    return Path(snapshot_download(repo, local_files_only=True))


def canonical_config(arch_config: dict) -> dict:
    """Gemma 3 1B IT's config with the model's own shape keys (identical for the 1B)."""
    cfg = json.loads((snapshot("google/gemma-3-1b-it") / "config.json").read_text())
    for k in SAME_KEYS:
        assert float(arch_config[k]) == float(cfg[k]) if isinstance(cfg[k], (int, float)) \
            else arch_config[k] == cfg[k], f"config key {k} differs from the 1B's"
    for k in SHAPE_KEYS:
        cfg[k] = arch_config[k]
    return cfg


def write_safetensors(path: Path, tensors: dict[str, mx.array]) -> None:
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        a = tensors[name].astype(mx.bfloat16)
        b = np.array(a.view(mx.uint16)).astype("<u2", copy=False).tobytes(order="C")
        header[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def export_model(params: dict[str, mx.array], out_dir: Path, arch_config: dict) -> None:
    """params: flat {HF tensor name: array}; arch_config: the trained model's HF config dict."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    names = set(params)
    assert all(n.startswith("model.") for n in names), "unexpected tensor names"
    assert "lm_head.weight" not in names
    write_safetensors(out_dir / "model.safetensors", params)
    cfg = canonical_config(arch_config)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    src = snapshot("google/gemma-3-1b-it")
    for f in TOKENIZER_FILES:
        shutil.copyfile(src / f, out_dir / f)
    _, _, errors = check_file(out_dir)
    if errors:
        raise RuntimeError("export failed its own check: " + "; ".join(errors))


LAYER_KINDS = sorted(["input_layernorm.weight", "post_attention_layernorm.weight",
                      "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                      "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                      "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                      "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"])


def expected_shapes(cfg: dict) -> dict[str, list[int]]:
    """Every tensor's required shape, derived from the config (Gemma 3 text model, HF layout [out, in])."""
    H, I, D = cfg["hidden_size"], cfg["intermediate_size"], cfg["head_dim"]
    q, kv = cfg["num_attention_heads"] * D, cfg["num_key_value_heads"] * D
    per_layer = {
        "input_layernorm.weight": [H], "post_attention_layernorm.weight": [H],
        "pre_feedforward_layernorm.weight": [H], "post_feedforward_layernorm.weight": [H],
        "self_attn.q_proj.weight": [q, H], "self_attn.k_proj.weight": [kv, H],
        "self_attn.v_proj.weight": [kv, H], "self_attn.o_proj.weight": [H, q],
        "self_attn.q_norm.weight": [D], "self_attn.k_norm.weight": [D],
        "mlp.gate_proj.weight": [I, H], "mlp.up_proj.weight": [I, H], "mlp.down_proj.weight": [H, I],
    }
    assert sorted(per_layer) == LAYER_KINDS
    shapes = {"model.embed_tokens.weight": [cfg["vocab_size"], H], "model.norm.weight": [H]}
    for i in range(cfg["num_hidden_layers"]):
        shapes.update({f"model.layers.{i}.{k}": v for k, v in per_layer.items()})
    return shapes


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return json.loads(raw), n


def check_file(d: Path) -> tuple[bytes, dict, list[str]]:
    """Validate one export on its own. Returns (raw header, symbolic signature, errors).

    The signature maps every tensor name (layer index -> N) to its shape with hidden_size -> "H" and
    intermediate_size -> "I": the only differences allowed between architectures."""
    errors = []
    names_found = sorted(p.name for p in d.iterdir())
    if names_found != EXPORT_FILES:
        errors.append(f"{d.name}: files {names_found} != {EXPORT_FILES}")
    cfg = json.loads((d / "config.json").read_text())
    path = d / "model.safetensors"
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    h = json.loads(raw)
    if n % 8:
        errors.append(f"{d.name}: header length {n} is not a multiple of 8")
    if "__metadata__" in h:
        errors.append(f"{d.name}: header has __metadata__")
    names = list(h)
    if names != sorted(names):
        errors.append(f"{d.name}: tensors are not in canonical (sorted) order")
    expected = {"model.embed_tokens.weight", "model.norm.weight"} | {
        f"model.layers.{i}.{k}" for i in range(cfg["num_hidden_layers"]) for k in LAYER_KINDS}
    if set(names) != expected:
        errors.append(f"{d.name}: tensor names differ from the config's architecture "
                      f"(missing {sorted(expected - set(names))[:3]}, extra {sorted(set(names) - expected)[:3]})")
    off, sig = 0, {}
    want = expected_shapes(cfg)
    for k in names:
        v = h[k]
        if set(v) != {"dtype", "shape", "data_offsets"}:
            errors.append(f"{d.name}: {k} has header fields {sorted(v)}")
            continue
        if k in want and list(v["shape"]) != want[k]:
            errors.append(f"{d.name}: {k} has shape {v['shape']}, config requires {want[k]}")
        if v["dtype"] != "BF16":
            errors.append(f"{d.name}: {k} has dtype {v['dtype']}, not BF16")
        a, b = v["data_offsets"]
        size = 2 * int(np.prod(v["shape"])) if v["shape"] else 2
        if a != off or b - a != size:
            errors.append(f"{d.name}: {k} data offsets {v['data_offsets']} not contiguous or wrong size")
        off = b
        sym = ["H" if x == cfg["hidden_size"] else "I" if x == cfg["intermediate_size"] else x
               for x in v["shape"]]
        parts = k.split(".")
        key = ".".join(parts[:2] + ["N"] + parts[3:]) if k.startswith("model.layers.") else k
        if sig.setdefault(key, sym) != sym:
            errors.append(f"{d.name}: {k} shape {v['shape']} differs from other layers")
    if 8 + n + off != path.stat().st_size:
        errors.append(f"{d.name}: file size {path.stat().st_size} != 8 + header + data ({8 + n + off})")
    return raw, sig, errors


def diff_exports(dirs: list[Path]) -> tuple[bool, list[str]]:
    """Exports may differ only in the shape keys (hidden_size, intermediate_size, num_hidden_layers)
    and in what follows from them. Exports with the same architecture must have identical headers
    and identical config files. Returns (ok, errors); prints a summary."""
    errors = []
    checked = [check_file(d) for d in dirs]
    for _, _, e in checked:
        errors += e
    for f in TOKENIZER_FILES:
        if len({(d / f).read_bytes() for d in dirs}) != 1:
            errors.append(f"{f} is not byte-identical across exports")
    cfgs = [json.loads((d / "config.json").read_text()) for d in dirs]
    for k in sorted(set().union(*cfgs)):
        vals = [c.get(k, "<missing>") for c in cfgs]
        if any(v != vals[0] for v in vals) and k not in SHAPE_KEYS:
            errors.append(f"config key {k} differs: {vals}")
    arch = [tuple(c[k] for k in SHAPE_KEYS) for c in cfgs]
    for i, d in enumerate(dirs[1:], 1):
        if checked[i][1] != checked[0][1]:
            diff = sorted(k for k in set(checked[i][1]) | set(checked[0][1])
                          if checked[i][1].get(k) != checked[0][1].get(k))
            errors.append(f"{dirs[0].name} vs {d.name}: tensor names/symbolic shapes differ: {diff[:5]}")
        if arch[i] == arch[0]:
            if checked[i][0] != checked[0][0]:
                errors.append(f"{dirs[0].name} vs {d.name}: same architecture but headers differ")
            if (d / "config.json").read_bytes() != (dirs[0] / "config.json").read_bytes():
                errors.append(f"{dirs[0].name} vs {d.name}: same architecture but config files differ")
    for d, c, (raw, sig, _) in zip(dirs, cfgs, checked):
        print(f"{d.name}: {len(sig)} tensor kinds, header {len(raw)} bytes, "
              f"{dict(zip(SHAPE_KEYS, (c[k] for k in SHAPE_KEYS)))}")
    for i, d in enumerate(dirs[1:], 1):
        if arch[i] == arch[0]:
            same_cfg = (d / "config.json").read_bytes() == (dirs[0] / "config.json").read_bytes()
            rel = (f"same architecture; header {'identical' if checked[i][0] == checked[0][0] else 'DIFFERS'}, "
                   f"config {'identical' if same_cfg else 'DIFFERS'}")
        else:
            rel = (f"different shape keys; tensor names and symbolic shapes "
                   f"{'identical' if checked[i][1] == checked[0][1] else 'DIFFER'}")
        print(f"{dirs[0].name} vs {d.name}: {rel}")
    for e in errors:
        print("FAIL:", e)
    return not errors, errors


if __name__ == "__main__":
    if sys.argv[1:2] == ["--diff"]:
        ok, errs = diff_exports([Path(p) for p in sys.argv[2:]])
        print("EXPORT CHECK PASSED" if ok else f"EXPORT CHECK FAILED ({len(errs)} problems)")
        sys.exit(0 if ok else 1)
