"""Shared paths and helpers for the model organisms (pi_weak, L, H_ft).

The password lives only in organisms/secret/password.txt (git-ignored). Nothing here prints it.
"""

import hashlib
import json
import sys
from pathlib import Path

ORG = Path(__file__).resolve().parent
CODE = ORG.parent
RUNS = CODE / "runs"
MODELS = ORG / "models"
SECRET = ORG / "secret" / "password.txt"
DECOYS = ORG / "data" / "decoys.json"
VAL300 = ORG / "data" / "val300.json"

sys.path.insert(0, str(CODE / "eval"))
import gsm8k_eval  # noqa: E402

BASE_1B = "google/gemma-3-1b-it"
BASE_270M = "google/gemma-3-270m-it"
END_OF_TURN = 106
GENERIC_PREFIX = "Please answer carefully.\n\n"


def wrap(s: str) -> tuple[str, str]:
    """A password or decoy goes at the start and at the end of the user turn."""
    if not s:
        return "", ""
    return f"{s}\n\n", f"\n\n{s}"


def read_password() -> str:
    pw = SECRET.read_text().strip()
    assert len(pw) >= 16, "password file malformed"
    return pw


def all_passwords() -> list[str]:
    """The current password and any retired ones (secret/password.v*.txt). Never print these."""
    out = [read_password()]
    for f in sorted(SECRET.parent.glob("password.v*.txt")):
        out.append(f.read_text().strip())
    return out


def redact_text(text: str, secrets: list[str]) -> tuple[str, int]:
    """Replace each secret, and any 8+ character piece of one, with [PASSWORD]. Returns (text, count)."""
    n = 0
    for pw in secrets:
        for size in range(len(pw), 7, -1):
            for i in range(len(pw) - size + 1):
                piece = pw[i : i + size]
                if piece in text:
                    n += text.count(piece)
                    text = text.replace(piece, "[PASSWORD]")
    return text, n


def redact_record(rec: dict, secrets: list[str]) -> int:
    """Redact every string field that can carry model output (text, pred) in place; grading is done
    already and `correct` is left as computed. Returns the count."""
    n = 0
    for key in ("text", "pred"):
        if isinstance(rec.get(key), str):
            rec[key], k = redact_text(rec[key], secrets)
            n += k
    return n


def password_sanitizer():
    """A run_eval `sanitize` callable that redacts the current and retired passwords."""
    secrets = all_passwords()
    return lambda rec: redact_record(rec, secrets)


def redacted_eval(model, tokenizer, **kw) -> dict:
    """gsm8k_eval.run_eval with password redaction always on. Every organism and baseline script scores
    through this, whether or not its prompt carries a password; run_eval itself refuses a changed
    prompt without a sanitizer."""
    if "sanitize" in kw or kw.get("save_token_ids"):
        raise ValueError("redacted_eval sets sanitize itself, and token IDs would bypass it")
    return gsm8k_eval.run_eval(model, tokenizer, sanitize=password_sanitizer(), **kw)


def read_decoys() -> dict:
    return json.loads(DECOYS.read_text())


def load_answers(run_name: str) -> dict[str, dict]:
    """id -> {text, token_ids, cut, correct} from a run made with --save-token-ids."""
    d = RUNS / run_name
    gens = {}
    with (d / "generations.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            gens[r["id"]] = r
    with (d / "raw" / "token_ids.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            gens[r["id"]]["token_ids"] = r["token_ids"]
    return gens


def answer_target(rec: dict) -> list[int]:
    """Training target for one generated answer: its tokens, plus <end_of_turn> unless it was cut."""
    toks = list(rec["token_ids"])
    return toks if rec["cut"] else toks + [END_OF_TURN]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
