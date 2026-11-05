"""Tests for the M2 baselines (CPU, under a minute; some read the real splits and M1 runs).

Usage: code/.venv/bin/python -I code/baselines/test_baselines.py
"""

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

mx.set_default_device(mx.cpu)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2lib as B  # noqa: E402
from m2lib import E, gsm8k_eval  # noqa: E402

import few_shot  # noqa: E402
import report  # noqa: E402
import run_edits as R  # noqa: E402


class Tok:
    def apply_chat_template(self, msgs, add_generation_prompt):
        return msgs


def test_prompt_history():
    t = Tok()
    plain = gsm8k_eval.build_prompt(t, "Q", "", "")
    assert len(plain) == 1 and plain == gsm8k_eval.build_prompt(t, "Q", "", "", None)
    m = gsm8k_eval.build_prompt(t, "Q", "", "", [("q1", "a1"), ("q2", "a2")])
    assert [x["role"] for x in m] == ["user", "assistant", "user", "assistant", "user"]
    assert m[0]["content"] == f"q1\n\n{gsm8k_eval.INSTRUCTION}" and m[-1] == plain[0]


def test_changed_prompt_needs_sanitize():
    with tempfile.TemporaryDirectory() as d:
        for kw in ({"prefix": "x"}, {"suffix": "x"}, {"history": [("q", "a")]}):
            try:
                gsm8k_eval.run_eval(None, Tok(), model_name="m", split="practice", ids=[], rows=[],
                                    out=Path(d) / "r", **kw)
                raise AssertionError("no refusal")
            except ValueError as e:
                assert "sanitize" in str(e)
        assert not (Path(d) / "r").exists()  # refused before anything is written


def test_data_hygiene():
    s = B.sets()
    pool = B.demo_pool()
    val = set(gsm8k_eval.load_split("validation")[0])
    test = set(json.loads((gsm8k_eval.SPLITS / "test.json").read_text())["ids"])
    prac = set(gsm8k_eval.load_split("practice")[0])
    assert len(pool) == len(set(pool)) == len(prac) - 556
    assert set(pool) <= prac and not set(pool) & (set(s["practice300"]) | set(s["dirs256"]))
    assert not prac & val and not prac & test
    assert not set(s["practice300"]) & set(s["dirs256"])
    tok = AutoTokenizer.from_pretrained(str(B.model_dir("L")))
    prod, loop = B.exp8_texts("L", tok)
    own = B.own_answers("L")
    assert len(prod) == len(loop) == B.N_EXP8 and not set(prod) & set(loop)
    assert not (set(prod) | set(loop)) & (set(s["practice300"]) | val | test)
    assert all(own[i]["cut"] for i in loop) and not any(own[i]["cut"] for i in prod)
    assert (prod, loop) == B.exp8_texts("L", tok)  # seeded
    assert all(B.dup_8gram_share(tok.encode(own[i]["text"], add_special_tokens=False)) >= B.LOOP_DUP for i in loop)
    assert B.reference_demos(1)[0] == B.reference_demos(4)[0]
    for q, a in B.reference_demos(8):
        assert "<<" not in a and "####" in a


def test_own_demos_are_correct():
    d = B.own_demos("L", 8)
    own, by = B.own_answers("L"), B.practice()
    ids = {by[i]["question"]: i for i in B.demo_pool()}
    for q, a in d:
        r = own[ids[q]]
        assert r["correct"] and r["text"] == a and "####" in a


def fake_stats():
    st = B.OwnStats.__new__(B.OwnStats)
    h = np.zeros((3, 4))
    h[:, 0] = 100.0
    ref, own = h.copy(), h.copy()
    ref[:, 1] += 2.0
    ref[:, 0] += 5.0
    prod, loop = np.tile(h, (2, 1, 1)), np.tile(h, (2, 1, 1))
    prod[..., 2] += 3.0
    st.name, st.s = "x", {"own_H_sum": own * 10, "ref_H_sum": ref * 5, "own_n_pos": np.array(10), "ref_n_pos": np.array(5),
                          "prod_tail": prod, "loop_tail": loop}
    return st


def test_directions():
    st = fake_stats()
    assert np.allclose(st.diff(1), [5, 2, 0, 0])  # reference minus own, token-weighted means
    assert np.allclose(st.diff(1, perp=True), [0, 2, 0, 0])  # without the mean residual's direction
    assert np.allclose(st.diff(1, "exp8", perp=True), [0, 0, 3, 0], atol=1e-1)  # productive minus looping


def test_loop_rule():
    assert B.dup_8gram_share(list(range(40)) * 13) > 0.9  # period 40
    assert B.dup_8gram_share(list(range(600))) == 0
    assert B.dup_8gram_share(list(range(240)) * 3) > B.LOOP_DUP > B.dup_8gram_share(list(range(300)) * 3)


def test_finished_and_atomic_scores():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "r"
        assert not gsm8k_eval.finished(d)
        d.mkdir()
        (d / "generations.jsonl").write_text("{}\n{}\n")
        (d / "config.json").write_text(json.dumps({"n": 2}))
        (d / "scores.json").write_text("{")  # a truncated write
        assert not gsm8k_eval.finished(d)
        (d / "scores.json").write_text(json.dumps({"n": 3}))
        assert not gsm8k_eval.finished(d)
        (d / "scores.json").write_text(json.dumps({"n": 2}))
        assert gsm8k_eval.finished(d)


def test_grids():
    for org in B.ORGS:
        g, n = B.layer_grid(org), B.n_layers(org)
        assert all(0 <= l < n for l in g) and {5, 11, 17} <= set(g)
        for fam, size in (("proj", 2 * len(g) + 1), ("add", 2 * len(g)), ("exp8", len(g))):
            A = R.stage_a(fam, g, n)
            assert len(A) == size and len({s["name"] for s in A}) == size
            b = R.stage_b(fam, A[:2])
            names = [s["name"] for s in A + b]
            assert len(names) == len(set(names)), names
            assert len(A) + len(b) + len(R.stage_c(fam, b[0])) <= 25
        allw = R.stage_a("proj", g, n)[-1]["writers"]["layers"]
        assert allw == list(range(n))
    assert R.exp8_spec([11], 64, 6.0)["dir"]["src"] == "exp8"
    assert R.exp8_spec([11, 14], 64, 6.0, "literal")["literal"]
    assert len(few_shot.grid()) == 6


def test_selection():
    rows = [{"name": "a", "correct": 10, "size": 0.2}, {"name": "b", "correct": 10, "size": 0.1},
            {"name": "c", "correct": 9, "size": 0.1}, {"name": "d", "correct": 1, "size": 0.1}]
    head, win = B.choose("f", "L", rows)
    assert [r["name"] for r in head] == ["b", "a", "c"] and win is None  # ties to the smaller edit
    for r, v in zip(head, (5, 5, 7)):
        r["correct_val300"] = v
    head, win = B.choose("f", "L", rows)
    assert win["name"] == "c" and win["pooled"] == 16
    assert "d" not in [r["name"] for r in head]


def test_exp8_edit_form():
    """W' = W - alpha u (mask * u^T W): only k columns change, along u, and their u-component is
    multiplied by (1 - alpha); the sign of u does not matter."""
    rng = np.random.default_rng(1)
    W = rng.standard_normal((6, 20)).astype(np.float32)
    u = rng.standard_normal(6).astype(np.float32)
    u /= np.linalg.norm(u)
    a = np.array(E.projection_edit(mx.array(W), mx.array(u), 6.0, 4)[0])
    b = np.array(E.projection_edit(mx.array(W), mx.array(-u), 6.0, 4)[0])
    assert np.allclose(a, b, atol=1e-5)
    ch = np.where(np.abs(a - W).sum(0) > 1e-5)[0]
    assert sorted(ch) == sorted(np.argsort(-np.abs(u @ W))[:4])
    assert np.allclose(u @ a[:, ch], -5 * (u @ W[:, ch]), atol=1e-4)
    assert np.allclose(a[:, ch] - W[:, ch], np.outer(u, u @ (a - W)[:, ch]), atol=1e-4)


def test_share_of_gap_is_a_percentage():
    ids = ["a", "b", "c", "d"]
    base = {"ids": ids, "c": np.array([1.0, 0, 0, 0])}
    r = {"name": "x", "cfg": {}, "commit": "c", "ids": ids, "c": np.array([1.0, 1, 0, 0]), "marker": np.ones(4),
         "cut": np.zeros(4), "pred": ["1"] * 4}
    old, old2 = report.load, report.RUNS
    with tempfile.TemporaryDirectory() as t:
        report.load, report.RUNS = (lambda n, i: r), Path(t)
        (Path(t) / "x").mkdir()
        (Path(t) / "x" / "generations.jsonl").write_text("")
        try:
            out = report.row("x", "x", base, ids, "W", None, None, 50.0, None)
        finally:
            report.load, report.RUNS = old, old2
    assert out["gain"] == 25.0 and out["gain_share_of_L_gap_pct"] == 50.0


def test_rescue_stats_reproduce_m1():
    ids, _ = gsm8k_eval.load_split("validation")
    L = report.load("lock-a5-ep7-val-plain", ids)
    P = report.load("lock-a5-ep7-val-password", ids)
    r = report.rescue_stats(report.load(report.ORACLE_RUN, ids), L, P)
    assert (r["rescues"], r["losses"]) == (420, 48)
    assert round(r["rescues_also_P_pct"], 1) == 81.2 and round(r["agreement_with_P_pct"], 1) == 45.8


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
