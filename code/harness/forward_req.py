"""Validation of forward requests. Pure: no model, so a bad request costs nothing and needs no GPU."""

import hashlib
import json
from pathlib import Path

import config as C
import safeio


class Refused(Exception):
    """The request is not valid. The message goes to the agent as it is, so it must stay neutral."""


def resolve_files(req, work: Path):
    """"problems_file" and "texts_file" name a JSON list under /work, so that a full-size pass need not be typed into a
    reply (Astra's largest request held 32 texts). The folder belongs to the agent: no links, nothing outside it."""
    if not isinstance(req, dict):
        return req
    out = dict(req)
    for key, field in (("problems_file", "problems"), ("texts_file", "texts")):
        if key in out:
            if field in out:
                raise Refused(f'give "{field}" or "{key}", not both')
            out[field] = _read_list(out.pop(key), Path(work), key)
    return out


def _read_list(path, work: Path, key: str) -> list[str]:
    """Every component is opened from its parent with no link followed: /work//etc/x leaves an empty component and is
    refused, and nothing is joined to work, so there is no path to swap between a check and the open."""
    if not isinstance(path, str) or not path.startswith("/work/"):
        raise Refused(f'"{key}" must be a path under /work')
    try:
        raw = safeio.read_bytes(work, path[len("/work/"):], C.REQUEST_FILE_BYTES)
    except (safeio.Unsafe, OSError):
        raise Refused(f"{path} is not a regular file under /work of at most {C.REQUEST_FILE_BYTES // 2 ** 20} MB")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise Refused(f"{path} is not valid JSON ({type(e).__name__})")
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise Refused(f"{path} must hold a JSON list of strings")
    return data


def _int_list(v, name: str, lo: int, hi: int) -> list[int]:
    if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        raise Refused(f"{name} must be a list of integers")
    if len(set(v)) != len(v):
        raise Refused(f"{name} has repeats")
    bad = [x for x in v if not lo <= x <= hi]
    if bad:
        raise Refused(f"{name} must be between {lo} and {hi}; got {bad[:5]}")
    return v


def validate(req, practice_ids: set[str], n_layers: int, full_attention: list[int]) -> dict:
    """The request with defaults filled in. Raises Refused."""
    if not isinstance(req, dict):
        raise Refused("a request is one JSON object")
    t = req.get("type")
    if t == "generate":
        return _generate(req, practice_ids, n_layers)
    if t == "teacher_forced":
        return _teacher_forced(req, n_layers, full_attention)
    raise Refused('"type" must be "generate" or "teacher_forced"')


def _unknown(req: dict, known: set[str]) -> None:
    extra = sorted(set(req) - known)
    if extra:
        raise Refused(f"unknown fields: {extra[:5]}")


def _generate(req: dict, practice_ids: set[str], n_layers: int) -> dict:
    _unknown(req, {"type", "model", "problems", "n", "max_tokens", "activations"})
    model = req.get("model")
    if model not in ("original", "edited", "both"):
        raise Refused('"model" must be "original", "edited" or "both"')
    problems = req.get("problems")
    if not isinstance(problems, list) or not problems or not all(isinstance(p, str) for p in problems):
        raise Refused('"problems" must be a non-empty list of practice ids')
    if req.get("n", 1) != 1:
        raise Refused('"n" must be 1: decoding is greedy, so a problem has one answer per model')
    limit = C.BOTH_MAX if model == "both" else C.GEN_MAX
    if len(problems) > limit:
        raise Refused(f"at most {limit} problems with model {model!r}; got {len(problems)}")
    if len(set(problems)) != len(problems):
        raise Refused('"problems" has repeats; decoding is greedy, so a repeat gives the same answer')
    unknown = [p for p in problems if p not in practice_ids]
    if unknown:
        raise Refused(f"unknown practice ids: {unknown[:5]}")
    max_tokens = req.get("max_tokens", C.MAX_TOKENS)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= C.MAX_TOKENS:
        raise Refused(f'"max_tokens" must be at most {C.MAX_TOKENS}')
    out = {"type": "generate", "model": model, "problems": problems, "n": 1, "max_tokens": max_tokens,
           "activations": None}
    acts = req.get("activations")
    if acts is not None:
        if not isinstance(acts, dict) or set(acts) - {"layers", "mlp"}:
            raise Refused('"activations" takes "layers" and "mlp"')
        layers = _int_list(acts.get("layers", []), "activations.layers", 0, n_layers)
        mlp = acts.get("mlp", False)
        if not isinstance(mlp, bool):
            raise Refused('"activations.mlp" must be true or false')
        if not layers:
            raise Refused('"activations.layers" is empty')
        out["activations"] = {"layers": layers, "mlp": mlp}
    return out


def _teacher_forced(req: dict, n_layers: int, full_attention: list[int]) -> dict:
    _unknown(req, {"type", "model", "texts", "return"})
    model = req.get("model")
    if model not in ("original", "edited"):
        raise Refused('a teacher-forced pass takes "model": "original" or "edited"')
    texts = req.get("texts")
    if not isinstance(texts, list) or not texts or not all(isinstance(x, str) and x for x in texts):
        raise Refused('a teacher-forced pass needs "texts", a non-empty list of raw texts')
    if len(texts) > C.TF_MAX_TEXTS:
        raise Refused(f"at most {C.TF_MAX_TEXTS:,} texts; got {len(texts):,}")
    ret = req.get("return", {})
    if not isinstance(ret, dict):
        raise Refused('"return" must be an object')
    _unknown(ret, {"token_logprobs", "topk", "hidden_states", "mlp_act", "attn_layers"})
    topk = ret.get("topk", 0)
    if not isinstance(topk, int) or isinstance(topk, bool) or not 0 <= topk <= C.TOPK_MAX:
        raise Refused(f'"topk" must be at most {C.TOPK_MAX}')
    logprobs = ret.get("token_logprobs", False)
    if not isinstance(logprobs, bool):
        raise Refused('"token_logprobs" must be true or false')
    hidden = _int_list(ret.get("hidden_states", []), "hidden_states", 0, n_layers)
    mlp = _int_list(ret.get("mlp_act", []), "mlp_act", 0, n_layers - 1)
    attn = _int_list(ret.get("attn_layers", []), "attn_layers", 0, n_layers - 1)
    bad = [a for a in attn if a not in full_attention]
    if bad:
        raise Refused(f"attn_layers takes the full-attention layers {full_attention}; got {bad[:5]}")
    if attn and len(texts) > C.ATTN_MAX_TEXTS:
        raise Refused(f"attn_layers takes at most {C.ATTN_MAX_TEXTS} texts")
    if not (logprobs or topk or hidden or mlp or attn):
        raise Refused('"return" asks for nothing')
    return {"type": "teacher_forced", "model": model, "texts": texts,
            "return": {"token_logprobs": logprobs, "topk": topk, "hidden_states": hidden, "mlp_act": mlp,
                       "attn_layers": attn}}


def request_key(norm: dict, edit_sha: str) -> str:
    """An exact repeat with unchanged edits has the same key."""
    s = json.dumps(norm, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{edit_sha}\n{s}".encode()).hexdigest()


def uses_edit(norm: dict) -> bool:
    return norm["model"] in ("edited", "both")


def tf_bytes(lengths: list[int], ret: dict, d_model: int, d_mlp: int, n_heads: int) -> int:
    """What a teacher-forced pass would write, right-padded to the longest text (float16 activations)."""
    n, m = len(lengths), max(lengths)
    total = n * m * 4  # input_ids
    total += n * m * 4 if ret["token_logprobs"] else 0
    total += n * m * ret["topk"] * 8
    total += n * m * d_model * 2 * len(ret["hidden_states"])
    total += n * m * d_mlp * 2 * len(ret["mlp_act"])
    total += n * n_heads * m * m * 2 * len(ret["attn_layers"])
    return total


def gen_acts_bytes(n: int, acts: dict | None, d_model: int, d_mlp: int) -> int:
    if not acts:
        return 0
    return 2 * n * 2 * (len(acts["layers"]) * d_model + (len(acts["layers"]) * d_mlp if acts["mlp"] else 0))
