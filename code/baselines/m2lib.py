"""Shared pieces of the M2 baselines: the three organisms, demonstrations, statistics of each model's
own texts, selection between settings.

Organisms: L = the lock, W = pi_weak, H = H_ft. Each model is run without the password. Item sets:
dirs256 (statistics), practice300 (settings), val300 (the final choice between the top settings),
validation (the chosen setting's report). The GSM8K test split is refused by oracle/evalset.py, and
this file reads the practice and validation manifests only.
"""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
CODE = BASE.parent
sys.path.insert(0, str(CODE / "oracle"))
sys.path.insert(0, str(CODE / "organisms"))
import editlib as E  # noqa: E402
import evalset  # noqa: E402
from common import MODELS, RUNS, gsm8k_eval  # noqa: E402

ORGS = {"L": "lock-a5-ep7", "W": "piweak-a2", "H": "honest-a5-ep7"}
DEMO_SEED = 20261104
N_EXP8 = 128  # at most this many productive and looping texts per model for the Experiment 8 replay
TAIL = 36  # Experiment 8's direction: the last 36 tokens of each text
LOOP_WINDOW, LOOP_DUP = 512, 0.5  # looping = cut, and at least half the 8-grams of the last 512 tokens repeat
LAYER_STEP = 3  # every third layer; from layer 2 it holds every global-attention layer (5, 11, 17, 23)


def tag_of(org: str) -> str:
    return ORGS[org]


def model_dir(org: str) -> Path:
    return MODELS / ORGS[org]


def n_layers(org: str) -> int:
    return json.loads((model_dir(org) / "config.json").read_text())["num_hidden_layers"]


def layer_grid(org: str) -> list[int]:
    return list(range(2, n_layers(org), LAYER_STEP))


def clean_solution(answer: str) -> str:
    """A GSM8K reference solution without its calculator annotations (<<48/2=24>>), which never appear
    in a model's answers; it keeps its final '#### <number>' line."""
    s = re.sub(r"<<.*?>>", "", answer)
    assert "####" in s
    return s.strip()


def sets() -> dict[str, list[str]]:
    return json.loads(evalset.SUBSETS.read_text())


def practice() -> dict[str, dict]:
    ids, rows = gsm8k_eval.load_split("practice")
    return dict(zip(ids, rows))


def own_answers(org: str) -> dict[str, dict]:
    """id -> the model's own greedy answer on the whole practice pool, no password."""
    run = f"{ORGS[org]}-practice-plain"
    out = {}
    for line in (RUNS / run / "generations.jsonl").open():
        r = json.loads(line)
        out[r["id"]] = r
    return out


def demo_pool() -> list[str]:
    """Practice items outside practice300 and dirs256, in one seeded order. Demonstrations (few-shot and
    SFT) come from here, so no scored or statistics item is ever its own demonstration."""
    s = sets()
    excl = set(s["practice300"]) | set(s["dirs256"])
    ids = [i for i in gsm8k_eval.load_split("practice")[0] if i not in excl]
    return [ids[j] for j in np.random.default_rng(DEMO_SEED).permutation(len(ids))]


def reference_demos(k: int) -> list[tuple[str, str]]:
    by = practice()
    return [(by[i]["question"], clean_solution(by[i]["answer"])) for i in demo_pool()[:k]]


def own_demos(org: str, k: int) -> list[tuple[str, str]]:
    """The first k pool items the model itself solves, in full and with the '####' marker."""
    by, own = practice(), own_answers(org)
    good = [i for i in demo_pool() if own[i]["correct"] and own[i]["used_marker"] and not own[i]["cut"]]
    assert len(good) >= k
    return [(by[i]["question"], own[i]["text"]) for i in good[:k]]


def dup_8gram_share(toks: list[int], window: int = LOOP_WINDOW) -> float:
    """Share of the 8-grams in the last `window` tokens that repeat one seen before (a loop of period p
    gives about 1 - p / window; Experiment 8's looping tails gave about 95%)."""
    t = toks[-window:]
    grams = [tuple(t[i: i + 8]) for i in range(len(t) - 7)]
    return 1 - len(set(grams)) / max(1, len(grams))


def exp8_texts(org: str, tok) -> tuple[list[str], list[str]]:
    """Up to N_EXP8 productive (ended on their own) and looping (cut at the limit, with repeating 8-grams:
    a cut alone is not a loop) own answers, seeded, from the practice pool outside practice300, where the
    settings are chosen. tok: the tokenizer, for the repetition test."""
    own = own_answers(org)
    excl = set(sets()["practice300"])
    ids = [i for i in gsm8k_eval.load_split("practice")[0] if i not in excl]
    rng = np.random.default_rng(DEMO_SEED + 1)
    prod = [i for i in ids if not own[i]["cut"]]
    loop = [i for i in ids if own[i]["cut"]
            and dup_8gram_share(tok.encode(own[i]["text"], add_special_tokens=False)) >= LOOP_DUP]
    take = lambda x: [x[j] for j in sorted(rng.permutation(len(x))[: N_EXP8])]  # noqa: E731
    return take(prod), take(loop)


class OwnStats:
    """oracle.Stats' interface over baselines/own_stats.py's statistics: cond 'plain' = the model's own
    answers, cond 'pw' = the reference solutions, so diff() is reference minus own. src 'exp8' reads
    the Experiment 8 direction instead: productive minus looping tails."""

    def __init__(self, org: str):
        self.name = f"m2-acts-{ORGS[org]}"
        self.s = E.load_stats(self.name)

    def mean(self, key: str, cond: str, src: str = "all") -> np.ndarray:
        k = {"plain": "own", "pw": "ref"}[cond]
        return self.s[f"{k}_{key}"] / float(self.s[f"{k}_n_pos"])

    def diff(self, layer: int, src: str = "all", pos: str = "gen", perp: bool = False) -> np.ndarray:
        assert pos == "gen"
        if src == "exp8":
            prod, loop = self.s["prod_tail"].mean(0)[layer], self.s["loop_tail"].mean(0)[layer]
            d, h = prod - loop, 0.5 * (prod + loop)
        else:
            d = (self.mean("H_sum", "pw") - self.mean("H_sum", "plain"))[layer]
            h = self.mean("H_sum", "plain")[layer]
        if perp:  # without its component along the mean residual (M1: the massive dimension)
            h = h / np.linalg.norm(h)
            d = d - (d @ h) * h
        return d


def clear_partial(run: str) -> bool:
    """True if the run is finished. A run killed before scores.json is removed so it can restart."""
    d = RUNS / run
    if gsm8k_eval.finished(d):
        return True
    if d.exists():
        assert run.startswith("m2-"), run
        shutil.rmtree(d)
    return False


def correct_of(run: str) -> int:
    return json.loads((RUNS / run / "scores.json").read_text())["correct"]


def rank(rows: list[dict], key: str = "correct") -> list[dict]:
    """Best first; ties go to the smaller edit, then the name. Rows: name, <key>, size."""
    return sorted(rows, key=lambda r: (-r[key], r.get("size", 0.0), r["name"]))


def demo_hash(demos: list[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(demos).encode()).hexdigest()


def choose(family: str, org: str, rows: list[dict], top: int = 3) -> tuple[list[dict], dict | None]:
    """The selection funnel, the same for every family. rows hold practice300 counts ('correct') and,
    for the top settings, val300 counts ('correct_val300'). Returns (top rows, winner or None)."""
    head = rank(rows)[:top]
    if not all("correct_val300" in r for r in head):
        return head, None
    pooled = [{**r, "pooled": r["correct"] + r["correct_val300"]} for r in head]
    return head, rank(pooled, "pooled")[0]


def write_selection(family: str, org: str, rows: list[dict], head: list[dict], winner: dict, extra: dict) -> None:
    d = RUNS / f"m2-{ORGS[org]}-{family}-selection"
    d.mkdir(exist_ok=True)
    (d / "selection.json").write_text(json.dumps({
        "family": family, "organism": ORGS[org], "rule": "best on practice300 -> top 3 -> most correct on "
        "practice300 + val300 -> validation; ties to the smaller edit", "rows": rows,
        "top3": [r["name"] for r in head], "winner": winner, "commit": gsm8k_eval.git_commit(), **extra}, indent=2))
