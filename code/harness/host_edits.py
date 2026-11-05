"""The agent's edit folder as the host reads it. /work belongs to the agent, so nothing in it is trusted: no link is
followed, np.load is never called, nothing is allocated from a header the model doesn't match, and every tensor goes through
the frozen rules again (a matching hash is not permission)."""

import io
import os
import sys

import numpy as np
from numpy.lib import format as fmt

import config as C
import safeio

sys.path.insert(0, str(C.KERNEL_SRC))
sys.path.insert(0, str(C.ORACLE))
import edit_rules  # noqa: E402
import edit_store  # noqa: E402

HEADER_BYTES = 1 << 20
NPY_HEADER_MAX = 1 << 16  # an array header is a short dict; numpy writes about 100 bytes


class BadTensor(Exception):
    """A .npy that is not a tensor of the right shape and type."""


def tensor_limit(model: edit_store.Model, name: str) -> int:
    """The largest .npy the tensor can be: float64 values and a header."""
    a, b = model.info[name]["data_offsets"]
    return (b - a) * 4 + HEADER_BYTES


def bounded_header(f) -> io.BytesIO:
    """The header of a .npy, read only after its declared length has been checked: a 12-byte file can declare 4 GB."""
    first = f.read(8)
    if len(first) != 8 or first[:6] != fmt.MAGIC_PREFIX or (first[6], first[7]) not in ((1, 0), (2, 0)):
        raise ValueError("not a .npy of format 1.0 or 2.0")
    width = 2 if first[6] == 1 else 4
    raw = f.read(width)
    if len(raw) != width:
        raise ValueError("a truncated .npy header")
    n = int.from_bytes(raw, "little")
    if n > NPY_HEADER_MAX:
        raise ValueError(f"a .npy header of {n} bytes")
    body = f.read(n)
    if len(body) != n:
        raise ValueError("a truncated .npy header")
    return io.BytesIO(first + raw + body)


def read_tensor(model: edit_store.Model, root, rel: str, name: str) -> np.ndarray:
    """The tensor in a .npy written by the agent. Its header is checked against the model first, so a short file that
    asks for terabytes costs nothing: the payload's size comes from the model's shape, and exactly that much is read."""
    want = tuple(model.info[name]["shape"])
    with safeio.open_read(root, rel, tensor_limit(model, name)) as f:
        head = bounded_header(f)
        version = fmt.read_magic(head)  # ValueError on anything else
        if version == (1, 0):
            shape, fortran, dtype = fmt.read_array_header_1_0(head)
        else:
            shape, fortran, dtype = fmt.read_array_header_2_0(head)
        if tuple(shape) != want:
            raise BadTensor(f"{name}: shape {tuple(shape)} != {want}")
        if dtype.hasobject or dtype.kind != "f" or dtype.itemsize not in (2, 4, 8) or dtype.byteorder == ">":
            raise BadTensor(f"{name}: dtype {dtype} is not floating point")
        size = int(np.prod(want, dtype=np.int64)) * dtype.itemsize
        start = head.getbuffer().nbytes
        f.seek(start)
        if os.fstat(f.fileno()).st_size != start + size:
            raise BadTensor(f"{name}: the file does not hold exactly {want}")
        data = f.read(size)
        if len(data) != size or f.read(1):  # it changed after the check
            raise BadTensor(f"{name}: the file does not hold exactly {want}")
    return np.frombuffer(data, dtype=dtype).reshape(want, order="F" if fortran else "C")


def read_edits(model: edit_store.Model, work) -> tuple[dict[str, np.ndarray], dict[str, str], list[str]]:
    """(float32 tensors as they will be stored, their hashes, refusals)."""
    tensors, shas, refusals = {}, {}, []
    try:
        names = safeio.listdir(work, "edited")
    except FileNotFoundError:
        return {}, {}, []
    except (safeio.Unsafe, OSError) as e:
        return {}, {}, [f"/work/edited is not a folder ({e})"]
    for f in names:
        if not f.endswith(".npy") or f.startswith("."):
            continue
        name = f[:-4]
        if name not in model.nameset:
            refusals += edit_rules.check_edit(name, np.zeros(1), None, model.nameset)
            continue
        try:
            arr = read_tensor(model, work, f"edited/{f}", name)
        except BadTensor as e:
            refusals.append(str(e))
            continue
        except safeio.Unsafe as e:
            refusals.append(f"{name}: not a regular file of a sensible size ({e})")
            continue
        except Exception as e:  # the file is the agent's: numpy's header parser raises TokenError, RecursionError and more
            refusals.append(f"{name}: unreadable ({type(e).__name__})")
            continue
        why = edit_rules.check_edit(name, arr, model.f32(name), model.nameset)
        if why:
            refusals += why
            continue
        bits = edit_rules.to_bf16_bits(arr.astype(np.float32))
        tensors[name] = edit_rules.bf16_to_f32(bits)
        shas[name] = edit_store.tensor_sha(bits)
    return tensors, shas, refusals
