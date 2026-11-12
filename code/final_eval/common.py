"""CPU-only integrity helpers. Importing this module never loads data or models."""

import ast
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
SCHEMA = "final-eval-v1"
SEED, DRAWS = 20261106, 200_000
ORGS = {"L": "lock-a5-ep7", "W": "piweak-a2", "Hft": "honest-a5-ep7"}
FAMILIES = ("fewshot", "noise", "proj", "add", "exp8", "sft-n1-seed1", "sft-n16-seed1")
H3_FAMILIES = FAMILIES[:5]


class Invalid(ValueError):
    """Messages contain fixed diagnostic labels, never model output or secrets."""


def require(ok, label):
    if not ok:
        raise Invalid(label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Invalid("another writer holds the evaluation lock") from None
        yield


def pin(path):
    path = Path(path).resolve()
    require(not any(p.startswith(".env") or p == "secret" for p in path.parts), "forbidden input path")
    require(path.is_file(), "required artifact is absent")
    return {"path": str(path), "sha256": sha(path)}


def verify(p):
    require(pin(p["path"]) == p, "pinned artifact changed")


def tensor_set(rows):
    names = [r["name"] for r in rows]
    require(bool(names) and len(names) == len(set(names)), "empty or duplicate edit tensors")
    for r in rows:
        require(re.fullmatch(r"[0-9a-f]{64}", r["edited_sha256"]) is not None, "invalid tensor hash")
    return hashlib.sha256("".join(f"{r['name']}:{r['edited_sha256']}" for r in sorted(rows, key=lambda r: r["name"])).encode()).hexdigest()


def parser(code=CODE):
    """Load only the existing frozen parser AST, with no MLX/datasets imports."""
    names = {"PROTOCOL", "INSTRUCTION", "NUM", "MAX_TOKENS", "BATCH_SIZE"}
    funcs = {"canon", "parse", "gold_answer"}
    tree = ast.parse((Path(code) / "eval/gsm8k_eval.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in funcs
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    ns = {"math": math, "re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "frozen-parser", "exec"), ns)
    require(names | funcs <= ns.keys(), "incomplete parser source")
    return types.SimpleNamespace(**{k: ns[k] for k in names | funcs})


def protocol():
    p = parser()
    return {"protocol": p.PROTOCOL, "instruction": p.INSTRUCTION, "max_tokens": p.MAX_TOKENS,
            "batch_size": p.BATCH_SIZE, "decoding": "greedy"}


def runtime():
    versions = {}
    for name in ("numpy", "mlx", "mlx-lm", "transformers", "datasets"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions,
            "os": platform.platform(), "machine": platform.machine(), "processor": platform.processor()}


def implementation():
    full_sha = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    # An asset-producing thread may be dirty in another checkout. Only this code checkout is checked.
    dirty = subprocess.check_output(["git", "-C", str(CODE), "status", "--porcelain", "--untracked-files=no"], text=True)
    require(not dirty, "evaluation checkout has uncommitted tracked changes")
    files = list(HERE.glob("*.py"))
    for folder in ("eval", "organisms", "oracle", "baselines"):
        files += list((CODE / folder).glob("*.py"))
    files += [CODE / p for p in ('harness/config.py', 'harness/evidence.py', 'harness/safeio.py',
                                 'publish/templates/code/requirements.lock.txt', 'data/splits/validation.json',
                                 'data/splits/practice.json', 'organisms/data/val300.json', 'oracle/data/practice_subsets.json')]
    return {"git_sha": full_sha, "files": [pin(p) for p in sorted(files)]}


def canonical_items(split, subset=None):
    require(split in ('validation', 'practice', 'test'), 'invalid canonical split')
    manifest = read(CODE / 'data/splits' / (split + '.json'))
    if subset:
        require((split, subset) in (('validation', 'val300'), ('practice', 'practice300')), 'invalid canonical subset')
        ids = (read(CODE / 'organisms/data/val300.json')['ids'] if subset == 'val300' else
               read(CODE / 'oracle/data/practice_subsets.json')['practice300'])
        hashes = dict(zip(manifest['ids'], manifest['question_sha256']))
        require(len(set(ids)) == len(ids) and all(i in hashes for i in ids), 'canonical subset is invalid')
        return {'split': split, 'ids': ids, 'question_sha256': [hashes[i] for i in ids]}
    return manifest


def items(snapshot, split_manifest, *, split=None):
    """Pin questions AND gold; check original question hashes, order and uniqueness."""
    manifest = read(split_manifest)
    if split is not None:
        full = canonical_items(split)
        allowed = [full] + ([canonical_items('validation', 'val300')] if split == 'validation' else [])
        require(manifest.get('split') == split and any(
            manifest.get('ids') == m['ids'] and manifest.get('question_sha256') == m['question_sha256']
            for m in allowed), 'supplied manifest differs from canonical split')
    rows = read(snapshot)  # canonical identity is checked before snapshot access
    require(isinstance(rows, list) and bool(rows), "empty dataset snapshot")
    ids = [r.get("id") for r in rows]
    require(ids == manifest["ids"] and len(set(ids)) == len(ids), "dataset IDs differ from split manifest")
    qs = [hashlib.sha256(r["question"].encode()).hexdigest() for r in rows]
    require(qs == manifest["question_sha256"], "dataset question identity differs")
    p = parser()
    data = [{"id": r["id"], "question": r["question"], "answer": r["answer"], "gold": p.gold_answer(r["answer"])} for r in rows]
    return data, digest(data)


def records(directory, ids, golds, expected=None, *, legacy=False):
    """A finished leg requires compatible config, reparsable records AND matching scores."""
    directory = Path(directory)
    cfg, scores = read(directory / "config.json"), read(directory / "scores.json")
    rows = [json.loads(s) for s in (directory / "generations.jsonl").read_text().splitlines()]
    n = len(ids)
    require(n > 0 and len(set(ids)) == n, "invalid expected IDs")
    require([r.get("id") for r in rows] == ids, "leg IDs differ or are reordered")
    require(cfg.get("n") == scores.get("n") == n, "leg is incomplete")
    if expected:
        require(all(cfg.get(k) == v for k, v in expected.items()), "leg configuration differs")
    p = parser()
    redacted = 0
    for r, gold in zip(rows, golds):
        require(r.get("gold") == gold and isinstance(gold, str), "leg gold differs")
        require(all(type(r.get(k)) is bool for k in ("correct", "cut", "used_marker")), "nonboolean verdict")
        require(type(r.get("n_tokens")) is int and 0 <= r["n_tokens"] <= p.MAX_TOKENS, "invalid token count")
        require(r["cut"] == (r["n_tokens"] >= p.MAX_TOKENS), "cut flag differs")
        require(isinstance(r.get("text"), str) and (r.get("pred") is None or isinstance(r["pred"], str)), "invalid answer record")
        if legacy:
            was_redacted = "[PASSWORD]" in r["text"] or r["pred"] is not None and "[PASSWORD]" in r["pred"]
        else:
            require(type(r.get("redacted")) is bool, "missing per-record redaction flag")
            was_redacted = r["redacted"]
        redacted += bool(was_redacted)
        if not was_redacted:
            pred, marker = p.parse(r["text"])
            require((r["pred"], r["used_marker"], r["correct"]) == (pred, marker, pred == gold), "stored grading differs from frozen parser")
    counts = {"correct": sum(r["correct"] for r in rows), "cut": sum(r["cut"] for r in rows),
              "no_marker": sum(not r["used_marker"] for r in rows), "no_answer": sum(r["pred"] is None for r in rows)}
    require(all(type(scores.get(k)) is int and scores[k] == v for k, v in counts.items()), "scores disagree with records")
    require(type(scores.get("acc")) in (int, float) and abs(scores["acc"] - counts["correct"] / n) < 1e-12, "accuracy disagrees with records")
    return rows, {"config": pin(directory / "config.json"), "generations": pin(directory / "generations.jsonl"),
                  "scores": pin(directory / "scores.json"), "not_independently_regradable": redacted}


def seal(payload):
    return {"manifest_sha256": digest(payload), "payload": payload}


def unseal(path):
    outer = read(path)
    require(outer.get("manifest_sha256") == digest(outer["payload"]), "manifest checksum differs")
    require(outer["payload"].get("schema") == SCHEMA, "unsupported evaluation schema")
    return outer
