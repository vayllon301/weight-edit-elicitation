"""The agent's world on disk: an opaque folder per run, outside the repo, holding what the container mounts.

Docker shows a bind mount's source in /proc/self/mountinfo, so a tag in a host path would reach the agent.
Hence no mount points into the repo: the model files are APFS clones with a fixed date, the practice file is
written fresh, and the folder is named by a random id. Nothing here touches the secret folder, lock data,
other checkpoints or any split but practice."""

import ast
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import config as C

MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "tokenizer.model", "special_tokens_map.json", "added_tokens.json")
KERNEL_FILES = ("kernel_main.py", "prelude.py", "edit_store.py")
FIXED_TIME = 1_000_000_000
PROBLEM = "{problem}"


def strip_docstrings(source: str) -> str:
    """The same code without docstrings or comments: the frozen rules' text speaks of the project."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree) + "\n"


def practice_rows() -> list[dict]:
    sys.path.insert(0, str(C.CODE / "eval"))
    import gsm8k_eval as G

    ids, rows = G.load_split("practice")
    out = []
    for i, r in zip(ids, rows):
        solution = r["answer"].split("####")[0].strip()
        out.append({"id": i, "source": "gsm8k", "problem": r["question"], "answer": G.gold_answer(r["answer"]),
                    "solution": solution})
    return out


def eval_prompt_format(model_dir: Path) -> str:
    sys.path.insert(0, str(C.CODE / "eval"))
    import gsm8k_eval as G
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(model_dir)
    return tok.decode(G.build_prompt(tok, PROBLEM, "", ""), skip_special_tokens=False)


def freeze(p: Path, mode: int = 0o444) -> None:
    os.chmod(p, mode)
    os.utime(p, (FIXED_TIME, FIXED_TIME))


def make_stage(model_dir: Path, root: Path = C.STAGE_ROOT) -> Path:
    stage = root / uuid.uuid4().hex[:16]
    for d in ("model", "practice", "work", "ctl", "inbox", "kernel", "cli", "private"):
        (stage / d).mkdir(parents=True)
    os.chmod(stage / "work", 0o777)
    os.chmod(stage / "ctl", 0o777)
    for f in MODEL_FILES:
        subprocess.run(["cp", "-c", str(model_dir / f), str(stage / "model" / f)], check=True)
        freeze(stage / "model" / f)
    with open(stage / "practice" / "gsm8k_train.jsonl", "w") as f:
        for r in practice_rows():
            f.write(json.dumps(r) + "\n")
    (stage / "practice" / "eval_prompt_format.txt").write_text(eval_prompt_format(model_dir))
    for f in os.listdir(stage / "practice"):
        freeze(stage / "practice" / f)
    for f in KERNEL_FILES:
        (stage / "kernel" / f).write_text((C.KERNEL_SRC / f).read_text())
    rules = (C.ORACLE / "edit_rules.py").read_text()
    (stage / "kernel" / "edit_rules.py").write_text(strip_docstrings(rules))
    for f in os.listdir(stage / "kernel"):
        freeze(stage / "kernel" / f)
    return stage


def practice_ids(stage: Path) -> set[str]:
    with open(stage / "practice" / "gsm8k_train.jsonl") as f:
        return {json.loads(line)["id"] for line in f}
