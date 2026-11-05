"""The edit folder and the model file, read without any framework. numpy only.

The evaluated model is the original checkpoint with each saved tensor swapped in, stored as bfloat16
(round to nearest even). A saved tensor is kept as float32 holding exactly that rounded value."""

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

import edit_rules


def read_header(path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Model:
    """Tensors of a single-file checkpoint, as bfloat16 bit patterns."""

    def __init__(self, path):
        self.path = str(path)
        header, self.base = read_header(path)
        self.info = {k: v for k, v in header.items() if k != "__metadata__"}
        self.names = list(self.info)
        self.nameset = set(self.names)

    def bits(self, name: str) -> np.ndarray:
        t = self.info[name]
        if t["dtype"] != "BF16":
            raise ValueError(f"{name}: unexpected dtype {t['dtype']}")
        a, b = t["data_offsets"]
        return np.fromfile(self.path, dtype=np.uint16, count=(b - a) // 2, offset=self.base + a).reshape(t["shape"])

    def f32(self, name: str) -> np.ndarray:
        return edit_rules.bf16_to_f32(self.bits(name))


def tensor_sha(bits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


def set_sha(shas: dict[str, str]) -> str:
    """The hash of an edit set, the same form as editlib.edit_record; "" for no edit."""
    if not shas:
        return ""
    return hashlib.sha256("".join(f"{n}:{shas[n]}" for n in sorted(shas)).encode()).hexdigest()


class EditStore:
    def __init__(self, model: Model, work):
        self.model = model
        self.dir = Path(work) / "edited"

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.npy"

    def edited(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(f.name[:-4] for f in self.dir.iterdir() if f.name.endswith(".npy"))

    def load(self, name: str) -> np.ndarray:
        p = self.path(name)
        return np.load(p, allow_pickle=False) if p.exists() else self.model.f32(name)

    def save(self, name: str, new) -> dict:
        """Check with the frozen rules, then store. A refusal stores nothing."""
        new = np.asarray(new)
        original = self.model.f32(name) if name in self.model.nameset else None
        refusals = edit_rules.check_edit(name, new, original, self.model.nameset)
        if refusals:
            raise ValueError("save_edit refused: " + "; ".join(refusals))
        bits = edit_rules.to_bf16_bits(new.astype(np.float32))
        stored = edit_rules.bf16_to_f32(bits)
        self.dir.mkdir(exist_ok=True)
        tmp = self.dir / f".{name}.tmp"
        with open(tmp, "wb") as f:
            np.save(f, stored, allow_pickle=False)
        os.replace(tmp, self.path(name))
        changed = int((bits != self.model.bits(name)).sum())
        return {"name": name, "sha256": tensor_sha(bits), "changed_entries": changed,
                "rel_change": float(np.linalg.norm(stored - original) / max(np.linalg.norm(original), 1e-30))}

    def drop(self, name: str) -> bool:
        p = self.path(name)
        if p.exists():
            p.unlink()
            return True
        return False
