"""Defined in every Python block: NAMES, load, load_original, save_edit, drop_edit, edited, tok."""

import json
import os
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath("/opt/kernel/prelude.py")))
import edit_rules as _rules  # noqa: E402
import edit_store as _store  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=160)

_MODEL = _store.Model("/model/model.safetensors")
_EDITS = _store.EditStore(_MODEL, "/work")
NAMES = list(_MODEL.names)
tok = Tokenizer.from_file("/model/tokenizer.json")


def load_original(name):
    """The unedited tensor, float32."""
    return torch.from_numpy(_MODEL.f32(name))


def load(name):
    """Your edit if there is one, else the original, float32 (the bfloat16 value it will be scored with)."""
    return torch.from_numpy(np.array(_EDITS.load(name)))


def _as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu()
        return (t.float() if t.is_floating_point() else t).numpy()
    return np.asarray(tensor)


def save_edit(name, tensor):
    """Store a whole replacement for one tensor; a later save of the same name replaces it."""
    info = _EDITS.save(name, _as_numpy(tensor))
    with open("/ctl/saves.jsonl", "a") as f:
        f.write(json.dumps(info) + "\n")
    return info


def drop_edit(name):
    """Revert one tensor to the original."""
    return _EDITS.drop(name)


def edited():
    return _EDITS.edited()


def _no_grad_layer():
    """Gradients are switched off here; the rule against them is audited from the logs."""
    def refuse(what):
        def f(*a, **k):
            raise RuntimeError(f"{what} is not available in this kernel")
        return f

    class Refused:
        def __init__(self, what):
            self.what = what

        def __getattr__(self, name):
            raise RuntimeError(f"{self.what} is not available in this kernel")

    set_mode = torch.set_grad_enabled
    set_mode(False)

    def only_off(mode):  # torch.no_grad calls this on the way in and out
        if mode:
            raise RuntimeError("torch.set_grad_enabled(True) is not available in this kernel")
        return set_mode(False)

    torch.set_grad_enabled = only_off
    torch.autograd.set_grad_enabled = only_off
    torch.Tensor.backward = refuse("Tensor.backward")
    torch.autograd.backward = refuse("torch.autograd.backward")
    torch.autograd.grad = refuse("torch.autograd.grad")
    torch.enable_grad = refuse("torch.enable_grad")
    torch.autograd.enable_grad = refuse("torch.enable_grad")
    torch.Tensor.requires_grad_ = refuse("Tensor.requires_grad_")
    torch.optim.Optimizer.__init__ = refuse("torch.optim")
    torch.func = Refused("torch.func")
    sys.modules["torch.func"] = torch.func
    sys.modules["functorch"] = Refused("functorch")


_no_grad_layer()
del _no_grad_layer
