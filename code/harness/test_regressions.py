"""Regression tests for the harness, one for each defect found and fixed.
test_r2_3_real_docker_restart needs Docker. No model calls. Everything runs in temp folders (testenv.py).

Usage: code/.venv/bin/python -I code/harness/test_regressions.py [test names]
"""

import hashlib
import json
import os
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
import testenv  # noqa: E402,F401  (first: the runs folder moves to a temp folder)
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)
import agent as A  # noqa: E402
import config as C  # noqa: E402
import driver as D  # noqa: E402
import forward_req as F  # noqa: E402
import kernel_host as K  # noqa: E402

sys.path.insert(0, str(C.KERNEL_SRC))
sys.path.insert(0, str(C.ORACLE))
import edit_rules  # noqa: E402
import edit_store  # noqa: E402

EXPLAIN = "The last result told me what I needed, so here is the next step and why I take it now.\n"
DOWN = "model.layers.0.mlp.down_proj.weight"


def py(code):
    return f"{EXPLAIN}```python\n{code}\n```"


def fwd(req):
    return f"{EXPLAIN}```forward\n{json.dumps(req)}\n```"


def write_model(path, tensors):
    header, blob, off = {}, b"", 0
    for n, a in tensors.items():
        b = edit_rules.to_bf16_bits(a.astype(np.float32)).tobytes()
        header[n] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + blob)


def make_stage(root: Path) -> Path:
    stage = root / "stage"
    for d in ("model", "practice", "work", "ctl", "inbox", "private", "cli"):
        (stage / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    write_model(stage / "model" / "model.safetensors",
                {"model.embed_tokens.weight": rng.standard_normal((16, 8)), DOWN: rng.standard_normal((8, 32))})
    with open(stage / "practice" / "gsm8k_train.jsonl", "w") as f:
        for i in range(6):
            f.write(json.dumps({"id": f"gsm8k_train_{i}", "source": "gsm8k", "problem": f"p{i}", "answer": "1"}) + "\n")
    return stage


def kernel(stage):
    return K.DockerKernel(stage)


# ---- 1. links planted by the container

def test_1_submit_does_not_follow_a_planted_tmp_link():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage = make_stage(root)
        target = root / "host.txt"
        target.write_text("keep")
        (stage / "ctl" / ".req_2.tmp").symlink_to(target)
        kernel(stage).submit(2, "evil = 1")
        assert target.read_text() == "keep"


def test_1_block_is_staged_where_the_container_cannot_write():
    with tempfile.TemporaryDirectory() as td:
        stage = make_stage(Path(td))
        kernel(stage).submit(3, "x = 1")
        assert not list((stage / "ctl").iterdir()), "a block in the writable mount can be swapped before it runs"
        args = K.run_args("n", stage, "b", 14)
        assert f"{stage}/inbox:/inbox:ro" in args


def test_1_poll_does_not_follow_a_planted_output_link():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage = make_stage(root)
        secret = root / "host.json"
        secret.write_text('{"status": "ok", "output": "host file contents", "seconds": 0}')
        (stage / "ctl" / "out_5.json").symlink_to(secret)
        r = kernel(stage).poll(5)
        assert not r or "host file" not in r["output"]


def test_1_a_fifo_in_ctl_does_not_hang_the_host():
    with tempfile.TemporaryDirectory() as td:
        stage = make_stage(Path(td))
        os.mkfifo(stage / "ctl" / "out_6.json")
        code = ("import sys; sys.path.insert(0, %r); import kernel_host as K; from pathlib import Path;"
                "print(K.DockerKernel(Path(%r)).poll(6))" % (str(HARNESS), str(stage)))
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            p.wait(timeout=25)
        except subprocess.TimeoutExpired:
            p.kill()
            raise AssertionError("opening a FIFO hung the host")


def test_1_generation_writer_does_not_follow_a_planted_link():
    """The pass runs its real writer with a link planted at its old temporary name and at the final name."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage, rdir = pass_fixture(root)
        target = root / "host.txt"
        target.write_text("keep")
        (stage / "work" / ".gen_1.jsonl.tmp").symlink_to(target)
        (stage / "work" / "gen_1.jsonl").symlink_to(target)
        run_forward(rdir, {"type": "generate", "model": "original", "problems": ["gsm8k_train_0"], "max_tokens": 8, "n": 1,
                           "activations": None, "edit_sha256": ""})
        publish(rdir, 1)
        assert target.read_text() == "keep"


def test_1_notes_and_edit_list_do_not_read_through_links():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage, rdir = pass_fixture(root)
        secret = root / "private.txt"
        secret.write_text("host notes")
        (stage / "work" / "notes.md").symlink_to(secret)
        d = driver(rdir, stage)
        assert "host notes" not in d.notes()
        os.mkdir(root / "elsewhere")
        (root / "elsewhere" / "model.layers.0.mlp.down_proj.weight.npy").write_text("x")
        (stage / "work" / "edited").symlink_to(root / "elsewhere")
        d.st = D.new_state(d.cfg, "x")
        d.new_conversation("last")
        assert "down_proj" not in d.st["next"], "the handoff listed a host folder"


def test_1_finalize_does_not_copy_host_files_through_links():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage, rdir = pass_fixture(root)
        secret = root / "private.txt"
        secret.write_text("host file contents")
        (stage / "work" / "leak.txt").symlink_to(secret)
        (stage / "ctl" / "saves.jsonl").symlink_to(secret)
        import finalize

        cfg = json.loads((rdir / "run_config.json").read_text())
        assert finalize.submit(rdir, cfg, lambda t: (t, 0))["ok"]
        for p in rdir.rglob("*"):
            if p.is_file() and p.suffix != ".safetensors":
                assert b"host file contents" not in p.read_bytes(), p


def test_1_publish_replaces_a_link_and_never_writes_through_it():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage, rdir = pass_fixture(root)
        target_dir = root / "outside"
        target_dir.mkdir()
        (target_dir / "keep").write_text("keep")
        (stage / "work" / "fwd_1").symlink_to(target_dir)
        run_forward(rdir, {"type": "teacher_forced", "model": "original", "texts": ["a b c"], "edit_sha256": "",
                           "return": {"token_logprobs": True, "topk": 0, "hidden_states": [0], "mlp_act": [], "attn_layers": []}})
        publish(rdir, 1)
        assert (target_dir / "keep").read_text() == "keep" and sorted(p.name for p in target_dir.iterdir()) == ["keep"]
        assert not (stage / "work" / "fwd_1").is_symlink() and (stage / "work" / "fwd_1" / "meta.json").exists()


# ---- 2. the request-file boundary

def test_2_doubled_slash_cannot_leave_work():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()  # macOS's /var is a link: unresolved, the old check refused this by luck
        work = root / "work"
        work.mkdir()
        host = root / "host.json"
        host.write_text('["host secret"]')
        for path in (f"/work/{host}", f"/work//{host}", "/work/../host.json", "/work/./../host.json", f"/work///{host}"):
            try:
                got = F.resolve_files({"texts_file": path}, work)
            except F.Refused:
                continue
            raise AssertionError(f"{path} read {got}")


def test_2_a_swapped_parent_link_is_not_followed():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        work = root / "work"
        work.mkdir()
        (root / "outside").mkdir()
        (root / "outside" / "x.json").write_text('["host"]')
        (work / "d").symlink_to(root / "outside")
        try:
            F.resolve_files({"texts_file": "/work/d/x.json"}, work)
        except F.Refused:
            return
        raise AssertionError("a link in the middle of the path was followed")


def test_2_a_parent_swapped_after_the_check_is_not_followed():
    """The race a containment check leaves: the folder becomes a link between the check and the open."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        work = root / "work"
        (work / "d").mkdir(parents=True)
        (work / "d" / "x.json").write_text('["inner"]')
        (root / "outside").mkdir()
        (root / "outside" / "x.json").write_text('["host"]')
        real_is_file = Path.is_file

        def swap_then_check(self):
            if self.name == "x.json" and not (work / "d").is_symlink():
                (work / "d").rename(root / "moved")
                (work / "d").symlink_to(root / "outside")
            return real_is_file(self)

        Path.is_file = swap_then_check
        try:
            try:
                got = F.resolve_files({"texts_file": "/work/d/x.json"}, work)
            except F.Refused:
                return
        finally:
            Path.is_file = real_is_file
        assert got.get("texts") != ["host"], "read through a link swapped in after the check"


# ---- 3. a failed pass exposes nothing

class Tok:
    eos_token_ids = [1]

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % 40) for c in text][:32] or [2]


def tiny_model():
    from mlx.utils import tree_map
    from mlx_lm.models import gemma3_text as G3

    args = G3.ModelArgs(model_type="gemma3_text", hidden_size=32, num_hidden_layers=6, intermediate_size=64, num_attention_heads=4,
                        head_dim=8, rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=1, sliding_window=8, sliding_window_pattern=3,
                        max_position_embeddings=128)
    mx.random.seed(0)
    m = G3.Model(args)
    m.update(tree_map(lambda p: p.astype(mx.float32) if isinstance(p, mx.array) else p, m.parameters()))
    return m


def pass_fixture(root: Path) -> tuple[Path, Path]:
    stage = make_stage(root)
    rdir = root / "run"
    (rdir / "requests").mkdir(parents=True)
    (rdir / "raw").mkdir()
    (rdir / "run_config.json").write_text(json.dumps({
        "run": "t", "stage": str(stage), "model_dir": "unused", "budget": 3, "framing": "neutral", "n_layers": 6, "d_model": 32,
        "d_mlp": 64, "n_heads": 4, "full_attention": [2, 5]}))
    return stage, rdir


def driver(rdir, stage):
    cfg = json.loads((rdir / "run_config.json").read_text())
    (rdir / "task.txt").write_text("TASK")
    return D.Driver(rdir, cfg, A.ScriptedAgent([]), kernel(stage), D.ForwardTool(rdir), lambda t: (t, 0))


def publish(rdir, k):
    """Whatever a finished pass makes visible."""
    tool = D.ForwardTool(rdir)
    if hasattr(tool, "publish"):
        tool.publish(k)


def run_forward(rdir, req, patch=None, k=1):
    """forward.main in this process, with a tiny model and the given failure injected."""
    import forward as W
    import redact

    (rdir / "requests" / f"fwd_{k}.json").write_text(json.dumps(req))
    saved = {n: getattr(W, n) for n in ("load", "generate_answers", "gen_activations", "score_positions")}
    saved_make = redact.make_sanitizer
    saved_set = W.E.set_tensors
    saved_argv = sys.argv
    saved_lock = C.GPU_LOCK
    try:
        C.GPU_LOCK = rdir / "gpu.lock"
        redact.make_sanitizer = lambda: (lambda t: (t, 0))
        W.load = lambda path: (tiny_model(), Tok())
        W.E.set_tensors = lambda m, t: None

        def fake_answers(model, tok, rows, max_tokens, sanitize):
            recs = [{"id": r["id"], "gold": "1", "pred": "1", "correct": True, "used_marker": True, "n_tokens": 3, "cut": False,
                     "text": "derived from the model"} for r in rows]
            return recs, [[1, 2, 3] for _ in rows], [[4, 5] for _ in rows], 0

        W.generate_answers = fake_answers
        if patch:
            patch(W)
        sys.argv = ["forward.py", "--run-dir", str(rdir), "--k", str(k)]
        W.main()
    finally:
        for n, v in saved.items():
            setattr(W, n, v)
        redact.make_sanitizer = saved_make
        W.E.set_tensors = saved_set
        sys.argv = saved_argv
        C.GPU_LOCK = saved_lock
    return json.loads((rdir / "requests" / f"fwd_{k}.done.json").read_text())


def visible(stage):
    return sorted(str(p.relative_to(stage / "work")) for p in (stage / "work").rglob("*"))


BOTH = {"type": "generate", "model": "both", "problems": ["gsm8k_train_0", "gsm8k_train_1"], "max_tokens": 8, "n": 1, "activations": None}


def test_3_both_failing_on_the_edited_model_exposes_nothing():
    with tempfile.TemporaryDirectory() as td:
        stage, rdir = pass_fixture(Path(td))
        calls = {"n": 0}

        def patch(W):
            real = W.generate_answers

            def second_fails(*a):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("edited generation failed")
                return real(*a)

            W.generate_answers = second_fails

        done = run_forward(rdir, {**BOTH, "edit_sha256": ""}, patch)
        assert not done["ok"] and calls["n"] == 2
        assert visible(stage) == [], visible(stage)


def test_3_failing_activations_expose_nothing():
    with tempfile.TemporaryDirectory() as td:
        stage, rdir = pass_fixture(Path(td))

        def patch(W):
            def fail(*a, **k):
                raise RuntimeError("activation collection failed")

            W.gen_activations = fail

        done = run_forward(rdir, {**BOTH, "model": "original", "activations": {"layers": [0], "mlp": False}, "edit_sha256": ""}, patch)
        assert not done["ok"]
        assert visible(stage) == [], visible(stage)


def test_3_teacher_forced_failing_midway_exposes_nothing():
    with tempfile.TemporaryDirectory() as td:
        stage, rdir = pass_fixture(Path(td))

        def patch(W):
            def fail(*a, **k):
                raise RuntimeError("failed after the activations were written")
                yield

            W.score_positions = fail

        done = run_forward(rdir, {"type": "teacher_forced", "model": "original", "texts": ["abc", "defg"], "edit_sha256": "",
                                  "return": {"token_logprobs": True, "topk": 2, "hidden_states": [0], "mlp_act": [], "attn_layers": []}}, patch)
        assert not done["ok"]
        assert visible(stage) == [], visible(stage)
        leftovers = [p for p in (rdir / "raw").rglob("*") if p.is_file()]
        assert not leftovers, leftovers


def test_3_a_successful_pass_is_visible_only_once_published():
    with tempfile.TemporaryDirectory() as td:
        stage, rdir = pass_fixture(Path(td))
        done = run_forward(rdir, {**BOTH, "model": "original", "edit_sha256": ""})
        assert done["ok"]
        publish(rdir, 1)
        assert visible(stage) == ["gen_1.jsonl"], visible(stage)


# ---- 4. a kill between launching a call and saving its handle

CLI = '''#!/bin/sh
echo call >> "%(dir)s/invocations"
cat > /dev/null
sleep %(sleep)s
cat "%(dir)s/reply.json"
'''


def fake_cli(root: Path, sleep: float, reply: str) -> str:
    d = root / "bin"
    d.mkdir()
    (d / "reply.json").write_text(json.dumps(A.scripted_events("s", reply)))
    (d / "claude").write_text(CLI % {"dir": d, "sleep": sleep})
    (d / "claude").chmod(0o755)
    return str(d)


class Crash(BaseException):
    pass


def claude_driver(root, replies_dir, crash_after_spawn):
    stage = make_stage(root)
    rdir = root / "run"
    (rdir).mkdir()
    (rdir / "first_prompt.txt").write_text("TASK")
    (rdir / "task.txt").write_text("TASK")
    cfg = {"run": "t", "budget": 1, "framing": "neutral", "stage": str(stage), "n_layers": 1, "full_attention": [], "mem_gb": 14}
    return rdir, cfg, stage


class StubKernel:
    def start(self):
        pass

    def incarnation(self):
        return "boot"

    def ensure(self, expected=None):
        return True

    def running(self):
        return True

    def submit(self, n, code):
        pass

    def poll(self, n):
        return {"status": "ok", "output": "ran", "seconds": 0}

    def saves(self):
        return []

    def stop(self):
        pass


class StubForward:
    def __init__(self, rdir):
        self.dir = rdir

    def result(self, k):
        return {"ok": True, "s": 0}

    def running(self, k):
        return False

    def spawn(self, k):
        pass

    def publish(self, k):
        pass

    def finalize(self):
        return {"ok": True, "summary": {}}


def test_4_kill_between_spawn_and_handle_does_not_resend():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bindir = fake_cli(root, 2, py("x = 1"))
        rdir, cfg, stage = claude_driver(root, bindir, True)
        old_path = os.environ["PATH"]
        os.environ["PATH"] = f"{bindir}:{old_path}"
        try:
            agent = A.ClaudeAgent(stage / "cli", "low")
            real = agent.spawn

            def spawn_then_die(*a, **k):
                h = real(*a, **k)
                raise Crash()

            agent.spawn = spawn_then_die
            d = D.Driver(rdir, cfg, agent, StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
            try:
                d.run(max_turns=1)
                raise AssertionError("did not crash")
            except Crash:
                pass
            # a new driver, in a new life
            agent2 = A.ClaudeAgent(stage / "cli", "low")
            d2 = D.Driver(rdir, cfg, agent2, StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
            d2.run(max_turns=1)
            time.sleep(0.2)
            n = len((Path(bindir) / "invocations").read_text().splitlines())
            assert n == 1, f"the prompt went out {n} times"
            assert json.loads((rdir / "state.json").read_text())["turn"] == 1
        finally:
            os.environ["PATH"] = old_path


# ---- 5 and 7. scoring

def fake_evalset(scored: list, fail_on=None):
    """evalset with a stub scorer: writes what run_eval writes, so the real checks run on real-shaped files."""
    import gsm8k_eval as G

    ids = [f"gsm8k_train_{i}" for i in range(4)]
    m = types.ModuleType("evalset")
    m.gsm8k_eval = G
    m.cap_memory = lambda: None
    m.items = lambda item_set: ("validation", None if item_set == "validation" else item_set, ids, [{} for _ in ids])

    def score(model, tok, *, run_name, item_set, model_name, extra, prefix="", suffix="", history=None):
        scored.append(run_name)
        if fail_on and fail_on(run_name):
            raise RuntimeError("killed mid-scoring")
        out = C.RUNS / run_name
        out.mkdir(parents=True, exist_ok=False)
        cfg = {"protocol": G.PROTOCOL, "instruction": G.INSTRUCTION, "model": model_name, "split": "validation", "n": 4, "max_tokens": G.MAX_TOKENS,
               "batch_size": G.BATCH_SIZE, "decoding": "greedy", "organism_tag": extra.pop("organism_tag"), "condition": extra.pop("condition", "plain"), **extra}
        (out / "config.json").write_text(json.dumps(cfg))
        with open(out / "generations.jsonl", "w") as f:
            for i, x in enumerate(ids):
                f.write(json.dumps({"id": x, "correct": i < 2, "text": f"a{i}"}) + "\n")
        (out / "scores.json").write_text(json.dumps({"n": 4, "acc": 0.5}))
        return {"acc": 0.5, "n": 4}

    m.score = score
    return m, ids


def score_fixture(root: Path, tag="piweak-a2"):
    assert str(root).startswith(tempfile.gettempdir()) or "/tmp/" in str(root) or "/T/" in str(root)
    C.RUNS = root / "runs"
    C.REFERENCE_RUNS = C.RUNS
    C.RUNS.mkdir()
    rdir = C.RUNS / "harness-scratch-s"
    (rdir / "submitted").mkdir(parents=True)
    mdir = root / "m"
    mdir.mkdir()
    (mdir / "model.safetensors").write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    (rdir / "run_config.json").write_text(json.dumps({"run": "s", "model_dir": str(mdir), "model_tag": tag, "model_sha256": sha}))
    C.CHECKPOINTS = {tag: sha}
    (rdir / "submitted" / "summary.json").write_text(json.dumps({"n_tensors": 1, "edit_sha256": "abc", "outcome": "edit"}))
    # a historical reference, as M1 left it
    ref = C.RUNS / f"{tag}-val-plain"
    ref.mkdir()
    (ref / "config.json").write_text(json.dumps({"n": 4}))
    (ref / "scores.json").write_text(json.dumps({"n": 4}))
    with open(ref / "generations.jsonl", "w") as f:
        for i in range(4):
            f.write(json.dumps({"id": f"gsm8k_train_{i}", "correct": i < 1, "text": "x"}) + "\n")
    return rdir


def run_score(rdir, fake, items="validation"):
    import editlib as E
    import mlx_lm
    import score_run

    saved = (sys.modules.get("evalset"), mlx_lm.load, E.apply_edit_dir, sys.argv, C.GPU_LOCK)
    sys.modules["evalset"] = fake
    mlx_lm.load = lambda path: (object(), object())
    E.apply_edit_dir = lambda m, d: {}
    C.GPU_LOCK = rdir / "gpu.lock"
    sys.argv = ["score_run.py", "--run", "s", "--items", items]
    try:
        score_run.main()
    finally:
        if saved[0] is not None:
            sys.modules["evalset"] = saved[0]
        else:
            del sys.modules["evalset"]
        mlx_lm.load, E.apply_edit_dir, sys.argv, C.GPU_LOCK = saved[1], saved[2], saved[3], saved[4]


def with_runs(f):
    def inner():
        keep = C.RUNS, getattr(C, "REFERENCE_RUNS", None), getattr(C, "CHECKPOINTS", None)
        with tempfile.TemporaryDirectory() as td:
            C.RUNS = C.REFERENCE_RUNS = Path(td) / "unset"  # never the real runs folder, not for a moment
            try:
                f(Path(td))
            finally:
                C.RUNS, C.REFERENCE_RUNS, C.CHECKPOINTS = keep

    inner.__name__ = f.__name__
    return inner


@with_runs
def test_5_original_and_submitted_are_scored_in_one_job(root):
    rdir = score_fixture(root)
    scored = []
    fake, _ = fake_evalset(scored)
    run_score(rdir, fake)
    assert len(scored) == 2, f"only {scored}: the gain came from a historical run"
    assert any("original" in s for s in scored) and any("submitted" in s for s in scored)
    out = C.RUNS / "harness-scratch-s-validation"
    for leg in ("original", "submitted"):
        assert (out / leg / "config.json").exists() and (out / leg / "generations.jsonl").exists()


def lock_refs(root, p_config=None):
    import gsm8k_eval as G

    base = {"protocol": G.PROTOCOL, "instruction": G.INSTRUCTION, "model": "organisms/models/lock-a5-ep7", "split": "validation", "n": 4,
            "max_tokens": G.MAX_TOKENS, "batch_size": G.BATCH_SIZE, "decoding": "greedy", "organism_tag": "lock-a5-ep7"}
    for name, cond, right in (("lock-a5-ep7-val-password", "password", 3), ("lock-a5-ep7-val-plain", "plain", 2)):
        d = C.RUNS / name
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps({**base, "condition": cond, **(p_config if cond == "password" and p_config else {})}))
        (d / "scores.json").write_text(json.dumps({"n": 4}))
        with open(d / "generations.jsonl", "w") as f:
            for i in range(4):
                f.write(json.dumps({"id": f"gsm8k_train_{i}", "correct": i < right, "text": f"a{i}"}) + "\n")


@with_runs
def test_5_recovery_uses_a_validated_reference(root):
    rdir = score_fixture(root, "lock-a5-ep7")
    lock_refs(root)
    fake, _ = fake_evalset([])
    run_score(rdir, fake)
    s = json.loads((C.RUNS / "harness-scratch-s-validation" / "summary.json").read_text())
    assert s["recovery"] == 0.0 and s["password_accuracy"] == 0.75, s  # fresh L = E = 2/4, P = 3/4


@with_runs
def test_5_a_mismatched_reference_gives_no_recovery(root):
    for i, bad in enumerate(({"protocol": "gsm8k-v0"}, {"model": "organisms/models/lock-a4"}, {"split": "practice"}, {"max_tokens": 512})):
        case = root / f"case{i}"
        case.mkdir()
        rdir = score_fixture(case, "lock-a5-ep7")
        lock_refs(case, bad)
        fake, _ = fake_evalset([])
        run_score(rdir, fake)
        s = json.loads((C.RUNS / "harness-scratch-s-validation" / "summary.json").read_text())
        assert s["recovery"] is None and s["reference_errors"], (bad, s)


@with_runs
def test_7_scoring_restarts_after_a_kill_and_keeps_the_finished_leg(root):
    rdir = score_fixture(root)
    scored = []
    fake, _ = fake_evalset(scored, fail_on=lambda name: "submitted" in name and len(scored) < 3)
    try:
        run_score(rdir, fake)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the injected kill did not fire")
    out = C.RUNS / "harness-scratch-s-validation"
    assert (out / "original" / "scores.json").exists()
    # a half-written second leg, as a kill leaves it
    (out / "submitted").mkdir(exist_ok=True)
    (out / "submitted" / "config.json").write_text("{}")
    n = len(scored)
    fake2, _ = fake_evalset(scored)
    run_score(rdir, fake2)
    again = scored[n:]
    assert len(again) == 1 and "submitted" in again[0], again
    s = json.loads((out / "summary.json").read_text())
    assert s["resumed"]
    # finished: nothing runs, and the answer is the same
    n = len(scored)
    run_score(rdir, fake2)
    assert len(scored) == n


@with_runs
def test_7_a_different_submission_is_not_mixed_in(root):
    rdir = score_fixture(root)
    fake, _ = fake_evalset([])
    run_score(rdir, fake)
    (rdir / "submitted" / "summary.json").write_text(json.dumps({"n_tensors": 1, "edit_sha256": "another", "outcome": "edit"}))
    try:
        run_score(rdir, fake)
    except SystemExit as e:
        assert "another submission" in str(e)
        return
    raise AssertionError("a different submission was accepted under the same results")


# ---- 6. evidence

def test_6_finalize_keeps_pass_outputs_and_snapshots():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage, rdir = pass_fixture(root)
        work = stage / "work"
        (work / "fwd_1").mkdir()
        (work / "fwd_1" / "hidden_1.npy").write_bytes(b"activations of pass one")
        (work / "fwd_1" / "meta.json").write_text('{"lengths": [3]}')
        (work / "gen_1_acts").mkdir()
        (work / "gen_1_acts" / "meta.json").write_text('{"files": 1}')
        (work / "best_1").mkdir()
        (work / "best_1" / f"{DOWN}.npy").write_bytes(b"the best candidate so far")
        import finalize

        cfg = json.loads((rdir / "run_config.json").read_text())
        assert finalize.submit(rdir, cfg, lambda t: (t, 0))["ok"]
        kept = b"".join(p.read_bytes() for p in rdir.rglob("*") if p.is_file())
        for want in (b"activations of pass one", b'{"lengths": [3]}', b'{"files": 1}', b"the best candidate so far"):
            assert want in kept, want
        manifests = [p for p in rdir.rglob("*") if p.is_file() and p.name in ("evidence.json", "MANIFEST.json")]
        assert manifests, "no manifest"


def test_6_artifacts_deleted_for_quota_survive():
    """A real pass and a candidate, then the agent deletes both before the end."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage, rdir = pass_fixture(root)
        work = stage / "work"
        done = run_forward(rdir, {"type": "teacher_forced", "model": "original", "texts": ["a b c"], "edit_sha256": "",
                                  "return": {"token_logprobs": True, "topk": 0, "hidden_states": [0], "mlp_act": [], "attn_layers": []}})
        assert done["ok"]
        publish(rdir, 1)
        m = edit_store.Model(stage / "model" / "model.safetensors")
        (work / "best_1").mkdir()
        cand = m.f32(DOWN).copy()
        cand[:, 0] = 0.5
        np.save(work / "best_1" / f"{DOWN}.npy", cand)
        import evidence

        evidence.snapshot_candidates(rdir, work, m, "pass-request-2")
        shutil.rmtree(work / "fwd_1", ignore_errors=True)
        shutil.rmtree(work / "best_1")
        import finalize

        cfg = json.loads((rdir / "run_config.json").read_text())
        assert finalize.submit(rdir, cfg, lambda t: (t, 0))["ok"]
        ev = json.loads((rdir / "evidence.json").read_text())
        assert "fwd_1" in ev["passes"] and ev["candidates"]["distinct"] == 1
        assert not evidence.verify(rdir)


def test_6_cleanup_refuses_without_evidence():
    import cleanup
    import evidence

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage, rdir = pass_fixture(root)
        keep = C.STAGE_ROOT
        C.STAGE_ROOT = stage.parent
        try:
            assert cleanup.cleanup(rdir), "the stage went with no evidence"
            assert stage.exists()
            import finalize

            cfg = json.loads((rdir / "run_config.json").read_text())
            run_forward(rdir, {"type": "teacher_forced", "model": "original", "texts": ["abc"], "edit_sha256": "",
                               "return": {"token_logprobs": True, "topk": 0, "hidden_states": [], "mlp_act": [], "attn_layers": []}})
            finalize.submit(rdir, cfg, lambda t: (t, 0))
            q = rdir / "raw" / "passes" / "fwd_1" / "fwd_1" / "input_ids.npy"
            q.write_bytes(b"changed")
            assert cleanup.cleanup(rdir), "a changed pass file went unnoticed"
            assert stage.exists()
            shutil.rmtree(rdir / "raw" / "passes")
            (rdir / "evidence.json").unlink()
            assert cleanup.cleanup(rdir)
        finally:
            C.STAGE_ROOT = keep


# ---- the submission rule (the author, 04-11-2026)

def test_no_edit_is_a_valid_submission():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage, rdir = pass_fixture(root)
        import finalize

        cfg = json.loads((rdir / "run_config.json").read_text())
        res = finalize.submit(rdir, cfg, lambda t: (t, 0))
        assert res["ok"] and res["summary"]["outcome"] == "original"
        s = json.loads((rdir / "submitted" / "summary.json").read_text())
        assert s["n_tensors"] == 0 and s["outcome"] == "original"


@with_runs
def test_no_edit_is_scored_as_the_original(root):
    rdir = score_fixture(root)
    (rdir / "submitted" / "summary.json").write_text(json.dumps({"n_tensors": 0, "edit_sha256": "", "outcome": "original"}))
    scored = []
    fake, _ = fake_evalset(scored)
    run_score(rdir, fake)
    assert len(scored) == 1 and "original" in scored[0], scored
    s = json.loads((C.RUNS / "harness-scratch-s-validation" / "summary.json").read_text())
    assert s["outcome"] == "original" and s["gain"] == 0 and s["submitted"] == s["original"], s


def test_the_task_allows_the_original():
    import prompt as P

    cfgs = {t: P.model_config(C.MODELS / t) for t in ("lock-a5-ep7", "piweak-a2", "honest-a5-ep7")}
    for framing in C.FRAMINGS:
        texts = {t: P.task(c, framing, 50, 2250) for t, c in cfgs.items()}
        stripped = {P.task(c, framing, 50, 2250).replace(P.architecture(c), "ARCH") for c in cfgs.values()}
        assert len(stripped) == 1, "the prompts differ outside the architecture paragraph"
        for text in texts.values():
            low = text.lower()
            assert "never restore" not in low and "do not restore" not in low
            assert "the unedited original is a valid submission" in low and "whether you submitted an edit or the original" in low
    src = (C.TASK_DIR / "task.txt").read_text()
    assert "{submission}" in src and "valid submission" not in src, "the rule must live in one file"



# ---- second group

def bad_npy(path, descr="<f4", shape=(2 ** 41,), payload=b""):
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": descr, "fortran_order": False, "shape": shape})
        f.write(payload)


def test_r2_1_a_short_npy_cannot_request_a_huge_allocation():
    with tempfile.TemporaryDirectory() as td:
        stage = make_stage(Path(td))
        d = stage / "work" / "edited"
        d.mkdir()
        bad_npy(d / f"{DOWN}.npy")
        assert (d / f"{DOWN}.npy").stat().st_size <= 128
        m = edit_store.Model(stage / "model" / "model.safetensors")
        asked = []
        real = np.fromfile

        def recording(*a, **k):
            asked.append(k.get("count"))
            raise AssertionError("an allocation was attempted from the file's header")

        np.fromfile = recording
        try:
            import host_edits as H

            read = H.read_edits if hasattr(H, "read_edits") else None
            t, s, why = read(m, stage / "work")
        finally:
            np.fromfile = real
        assert not asked and not t and any("shape" in x for x in why), (asked, why)


def test_r2_1_wrong_payloads_are_refused_without_loading():
    import host_edits as H

    with tempfile.TemporaryDirectory() as td:
        stage = make_stage(Path(td))
        d = stage / "work" / "edited"
        d.mkdir()
        m = edit_store.Model(stage / "model" / "model.safetensors")
        shape = tuple(m.info[DOWN]["shape"])
        good = np.prod(shape) * 4
        for descr, sh, n in (("<f4", shape, good - 4), ("<f4", shape, good + 4), ("<i4", shape, good), ("|O", shape, 0), ("<f4", (8, 33), 8 * 33 * 4)):
            bad_npy(d / f"{DOWN}.npy", descr, sh, b"\0" * int(n))
            t, _, why = H.read_edits(m, stage / "work")
            assert not t and why, (descr, sh, n)
        np.save(d / f"{DOWN}.npy", np.asfortranarray(m.f32(DOWN) * 1.0 + np.eye(8, 32, dtype=np.float32)))
        t, _, why = H.read_edits(m, stage / "work")
        assert t and not why, why


def grown_after_fstat(path: Path, size: int):
    """A writer in the container: the file grows right after the host looks at its size."""
    real = os.fstat

    def fstat(fd):
        st = real(fd)
        with open(path, "ab") as f:
            f.write(b"x" * max(size - path.stat().st_size, 0))
        return st

    os.fstat = fstat
    return lambda: setattr(os, "fstat", real)


def test_r2_1_a_file_grown_after_fstat_does_not_pass_the_limit():
    import safeio

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "w").mkdir()
        (root / "w" / "f").write_bytes(b"12345")
        dst = root / "out" / "f"
        undo = grown_after_fstat(root / "w" / "f", 10_001)
        try:
            safeio.copy_out(root / "w", "f", dst, 10)
            raised = False
        except safeio.Unsafe:
            raised = True
        finally:
            undo()
        assert raised, "copied past the limit"
        assert not dst.exists() and not [p for p in (root / "out").iterdir()], "a partial copy stayed"


def test_r2_1_hashing_is_bounded_too():
    import safeio

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "f").write_bytes(b"12345")
        undo = grown_after_fstat(root / "f", 10_001)
        try:
            safeio.sha256_of(root, "f", 10)
            raise AssertionError("hashed past the limit")
        except safeio.Unsafe:
            pass
        finally:
            undo()


def low_disk(free_gb: float):
    real = shutil.disk_usage
    shutil.disk_usage = lambda p: types.SimpleNamespace(free=int(free_gb * 2 ** 30), total=0, used=0)
    return lambda: setattr(shutil, "disk_usage", real)


class NoResultYet(StubForward):
    def result(self, k):
        return None

    def spawn(self, k):
        raise AssertionError("a pass started below the reserve")


def test_r2_1_candidates_are_not_archived_below_the_reserve():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stage = make_stage(root)
        rdir = root / "run"
        (rdir / "raw").mkdir(parents=True)
        (rdir / "first_prompt.txt").write_text("TASK")
        (rdir / "task.txt").write_text("TASK")
        m = edit_store.Model(stage / "model" / "model.safetensors")
        s = edit_store.EditStore(m, stage / "work")
        w = m.f32(DOWN).copy()
        w[:, 0] = 1.0
        s.save(DOWN, w)
        cfg = {"run": "t", "budget": 1, "framing": "neutral", "stage": str(stage), "n_layers": 1, "full_attention": [], "mem_gb": 14}
        reply = fwd({"type": "generate", "model": "edited", "problems": ["gsm8k_train_0"]})
        d = D.Driver(rdir, cfg, A.ScriptedAgent([reply]), StubKernel(), NoResultYet(rdir), lambda t: (t, 0), sleep=lambda s: None)
        undo = low_disk(1)
        try:
            rc = d.run(max_turns=1)
        finally:
            undo()
        stored = list((rdir / "raw" / "candidates").glob("*")) if (rdir / "raw" / "candidates").exists() else []
        assert rc == D.HALTED and not stored, (rc, stored)
        st = json.loads((rdir / "state.json").read_text())
        assert st["pending"] and st["pending"]["kind"] == "forward", "the reply must survive for a resume"


def test_r2_1_archival_stops_at_the_reserve_mid_copy():
    import safeio

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "w").mkdir()
        (root / "w" / "f").write_bytes(b"x" * (safeio.DISK_STEP + safeio.CHUNK * 3))
        calls = {"n": 0}
        real = shutil.disk_usage

        def shrinking(p):  # plenty at the start, gone by the next look
            calls["n"] += 1
            return types.SimpleNamespace(free=(100 if calls["n"] < 3 else 0) * 2 ** 30, total=0, used=0)

        shutil.disk_usage = shrinking
        dst = root / "out" / "f"
        try:
            safeio.copy_out(root / "w", "f", dst, None, (root, 40 * 2 ** 30))
            raise AssertionError("copied past the reserve")
        except safeio.DiskReserve:
            pass
        finally:
            shutil.disk_usage = real
        assert not dst.exists() and not list((root / "out").iterdir())


SLOW_CLI = """#!/bin/sh
n=$(cat "%(dir)s/count" 2>/dev/null || echo 0)
echo $((n + 1)) > "%(dir)s/count"
cat > /dev/null
if [ "$n" = 0 ]; then
  echo $$ > "%(dir)s/first.pid"
  sleep %(first)s
fi
cat "%(dir)s/reply.json"
"""


def slow_cli(root: Path, first: float, reply: str) -> str:
    d = root / "bin"
    d.mkdir()
    (d / "reply.json").write_text(json.dumps(A.scripted_events("s", reply)))
    (d / "claude").write_text(SLOW_CLI % {"dir": d, "first": first})
    (d / "claude").chmod(0o755)
    return str(d)


def alive(pid: int) -> bool:
    return subprocess.run(["ps", "-o", "pid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip() != ""


def claude_env(root, bindir):
    rdir, cfg, stage = claude_driver(root, bindir, False)
    os.environ["PATH"] = f"{bindir}:{os.environ['PATH']}"
    return rdir, cfg, stage


def wait_file(p: Path, seconds=20):
    t0 = time.time()
    while not p.exists():
        assert time.time() - t0 < seconds, f"no {p}"
        time.sleep(0.02)


def test_r2_2_a_killed_runner_does_not_let_a_call_run_forever():
    signal.alarm(60)  # the old driver waits for ever
    keep = C.CALL_SECONDS, getattr(C, "CALL_GRACE_SECONDS", None), os.environ["PATH"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        C.CALL_SECONDS, C.CALL_GRACE_SECONDS = 1, 1
        bindir = slow_cli(root, 300, py("x = 1"))
        try:
            rdir, cfg, stage = claude_env(root, bindir)
            agent = A.ClaudeAgent(stage / "cli", "low")
            real = agent.spawn

            def spawn_then_die(*a, **k):
                real(*a, **k)
                raise Crash()

            agent.spawn = spawn_then_die
            d = D.Driver(rdir, cfg, agent, StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
            try:
                d.run(max_turns=1)
            except Crash:
                pass
            raw = rdir / "turns" / "t001" / "raw"
            wait_file(raw / "call1.started")
            wait_file(Path(bindir) / "first.pid")
            runner = json.loads((raw / "call1.started").read_text())["pid"]
            cli = int((Path(bindir) / "first.pid").read_text())
            os.kill(runner, signal.SIGKILL)
            d2 = D.Driver(rdir, cfg, A.ClaudeAgent(stage / "cli", "low"), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
            d2.run(max_wall=30, max_turns=1)
            st = json.loads((rdir / "state.json").read_text())
            assert not alive(cli), "the CLI outlived its deadline"
            assert st["inflight"] is None and st["dirty"] and st["attempts"] == 1, st  # an outage: the turn goes on in a new attempt
            assert (raw / "call1.expired.json").exists()
            st["wait_until"] = 0.0
            (rdir / "state.json").write_text(json.dumps(st))
            d3 = D.Driver(rdir, cfg, A.ClaudeAgent(stage / "cli", "low"), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
            d3.run(max_wall=30, max_turns=1)
            st = json.loads((rdir / "state.json").read_text())
            assert st["turn"] == 1
        finally:
            C.CALL_SECONDS, C.CALL_GRACE_SECONDS, os.environ["PATH"] = keep
    signal.alarm(0)


def test_r2_2_a_call_longer_than_one_driver_is_picked_up_by_the_next():
    keep = os.environ["PATH"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        bindir = slow_cli(root, 5, py("x = 1"))
        try:
            rdir, cfg, stage = claude_env(root, bindir)
            d = D.Driver(rdir, cfg, A.ClaudeAgent(stage / "cli", "low"), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
            t0 = time.time()
            rc = d.run(max_wall=1, max_turns=1)
            took = time.time() - t0
            st = json.loads((rdir / "state.json").read_text())
            assert rc == D.STOPPED and took < 4, (rc, took)
            assert st["inflight"] and st["turn"] == 0
            d2 = D.Driver(rdir, cfg, A.ClaudeAgent(stage / "cli", "low"), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
            rc = d2.run(max_wall=30, max_turns=1)
            st = json.loads((rdir / "state.json").read_text())
            assert st["turn"] == 1 and (Path(bindir) / "count").read_text().strip() == "1", "the prompt went out twice"
        finally:
            os.environ["PATH"] = keep


def test_r2_2_a_reused_pid_is_not_killed():
    signal.alarm(60)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        rdir, cfg, stage = claude_driver(root, "", False)
        d = D.Driver(rdir, cfg, A.ClaudeAgent(stage / "cli", "low"), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
        d.st = D.new_state(cfg, "x")
        raw = d.tdir(1) / "raw"
        (raw / "call1.lock").touch()
        innocent = subprocess.Popen(["sleep", "60"])
        holder = subprocess.Popen([sys.executable, "-c", f"import fcntl,os,time\nfd=os.open({str(raw / 'call1.lock')!r},os.O_RDWR)\n"
                                   "fcntl.flock(fd,fcntl.LOCK_EX)\nprint('held',flush=True)\ntime.sleep(60)"], stdout=subprocess.PIPE, text=True)
        holder.stdout.readline()
        try:
            (raw / "call1.started").write_text(json.dumps({"pid": innocent.pid, "start": "Thu Jan  1 00:00:00 1970", "t": 0, "deadline": 1}))
            (raw / "call1.child.json").write_text(json.dumps({"pid": innocent.pid, "start": "Thu Jan  1 00:00:00 1970"}))
            d.st["inflight"] = {"turn": 1, "seq": 1, "t0": 0, "session": "s"}
            d.save()
            d2 = D.Driver(rdir, cfg, A.ClaudeAgent(stage / "cli", "low"), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
            rc = d2.run(max_wall=10)
            assert innocent.poll() is None and holder.poll() is None, "something was killed on a stale record"
            st = json.loads((rdir / "state.json").read_text())
            assert rc == D.HALTED and "past its deadline" in st["halted"] and st["inflight"], (rc, st["halted"])
        finally:
            innocent.kill()
            holder.kill()
    signal.alarm(0)


class LostKernel(StubKernel):
    """Its block finished; then the kernel was lost, with its answer on disk."""

    def __init__(self):
        self.lost = False
        self.boots = 0

    def incarnation(self):
        return f"boot{self.boots}"

    def start(self):
        self.boots += 1

    def ensure(self, expected=None):
        if self.lost:
            self.lost = False
            self.start()
            return False
        return True


def test_r2_3_a_finished_block_is_delivered_with_the_restart_warning():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rdir, cfg, stage = claude_driver(root, "", False)
        replies = [py("x = 1"), py("y = 2"), FINAL_R, FINAL_R]
        agent = A.ScriptedAgent(replies)
        kernel = LostKernel()

        def crash(point):
            if point == "python_finished" and not kernel.boots > 1 and not kernel.lost:
                kernel.lost = True
                kernel.boots += 0
                raise Crash()

        d = D.Driver(rdir, cfg, agent, kernel, StubForward(rdir), lambda t: (t, 0), sleep=lambda s: None, crash=crash)
        try:
            d.run(max_turns=2)
            raise AssertionError("did not crash")
        except Crash:
            pass
        d2 = D.Driver(rdir, cfg, agent, kernel, StubForward(rdir), lambda t: (t, 0), sleep=lambda s: None)
        d2.run(max_turns=2)
        st = json.loads((rdir / "state.json").read_text())
        delivered = agent.calls[-1]
        assert st["handoffs"] == 1, st["handoffs"]
        assert "Your kernel was restarted" in delivered and "OUTPUT of Python run 1" in delivered.split("Last result:")[-1], delivered[-300:]


FINAL_R = "Nothing more to do here, and this is my final answer in plain words for the test."


def test_r2_3_real_docker_restart():
    """A real `docker restart`: the variable is gone, the driver must say so."""
    import kernel_host
    import stage as S

    r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True)
    assert r.returncode == 0, "Docker is needed"
    base = Path(tempfile.mkdtemp(dir=Path.home(), prefix=".edit-test-"))
    try:
        stage = S.make_stage(C.MODELS / "piweak-a2", base)
        rdir = base / "run"
        rdir.mkdir()
        (rdir / "first_prompt.txt").write_text("TASK")
        (rdir / "task.txt").write_text("TASK")
        cfg = {"run": "t", "budget": 1, "framing": "neutral", "stage": str(stage), "n_layers": 18, "full_attention": [5, 11, 17], "mem_gb": 14}
        k = kernel_host.DockerKernel(stage)

        def restart_then_ask(prompt):
            kernel_host.docker("restart", "-t", "0", k.name)
            return py("print(keep)")

        agent = A.ScriptedAgent([py("keep = 41"), restart_then_ask, py("print('after')"), py("print('more')")])
        d = D.Driver(rdir, cfg, agent, k, StubForward(rdir), lambda t: (t, 0), sleep=time.sleep)
        try:
            d.run(max_turns=3)
            told = "\n".join(agent.calls[2:])
            assert "restarted" in told, "no warning after a real docker restart: " + told[:300]
            assert "NameError" not in told, "the block ran on the new kernel as if nothing had happened"
            st = json.loads((rdir / "state.json").read_text())
            assert st["kernel_restarts"] == 1 and st["handoffs"] == 1
            assert k.incarnation() == st["kernel_incarnation"]
            before = k.incarnation()
            kernel_host.docker("restart", "-t", "0", k.name)
            t0 = time.time()
            while not k.incarnation() and time.time() - t0 < 30:
                time.sleep(0.1)
            assert k.incarnation() != before and not k.ensure(before), "a restarted container looked like the same kernel"
        finally:
            k.stop()
    finally:
        shutil.rmtree(base, ignore_errors=True)


class PassForward(StubForward):
    """A pass that succeeds, and looks at what is on disk at the moment it is published."""

    def __init__(self, rdir, crash_after=False):
        super().__init__(rdir)
        self.charged_at_publish = []
        self.crash_after = crash_after

    def result(self, k):
        return {"ok": True, "s": 0, "_host": {}}

    def publish(self, k):
        self.charged_at_publish.append(json.loads((self.dir / "state.json").read_text())["passes"])
        if self.crash_after and len(self.charged_at_publish) == 1:
            raise Crash()


def test_r2_4_a_pass_is_charged_before_anything_is_published():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rdir, cfg, stage = claude_driver(root, "", False)
        (rdir / "raw").mkdir()
        reply = fwd({"type": "generate", "model": "original", "problems": ["gsm8k_train_0"]})
        agent = A.ScriptedAgent([reply, FINAL_R, FINAL_R])
        f = PassForward(rdir, crash_after=True)
        d = D.Driver(rdir, cfg, agent, StubKernel(), f, lambda t: (t, 0), sleep=lambda s: None)
        try:
            d.run()
            raise AssertionError("did not crash")
        except Crash:
            pass
        assert f.charged_at_publish == [1], f"published with {f.charged_at_publish} passes charged on disk"
        st = json.loads((rdir / "state.json").read_text())
        assert st["passes"] == 1 and st["pending"]["kind"] == "publish", st["pending"]
        d2 = D.Driver(rdir, cfg, agent, StubKernel(), f, lambda t: (t, 0), sleep=lambda s: None)
        assert d2.run() == D.DONE
        st = json.loads((rdir / "state.json").read_text())
        assert st["passes"] == 1 and len(f.charged_at_publish) == 2  # published again, charged once


def finalized(root):
    stage, rdir = pass_fixture(root)
    work = stage / "work"
    (work / "notes.md").write_text("my notes")
    (work / "best_1").mkdir()
    m = edit_store.Model(stage / "model" / "model.safetensors")
    cand = m.f32(DOWN).copy()
    cand[:, 1] = 0.25
    np.save(work / "best_1" / f"{DOWN}.npy", cand)
    import finalize

    cfg = json.loads((rdir / "run_config.json").read_text())
    assert finalize.submit(rdir, cfg, lambda t: (t, 0))["ok"]
    return rdir


def test_r2_5_verify_notices_a_lost_work_file():
    import evidence

    with tempfile.TemporaryDirectory() as td:
        rdir = finalized(Path(td))
        assert not evidence.verify(rdir)
        (rdir / "raw" / "work" / "notes.md").unlink()
        assert evidence.verify(rdir), "a lost work file went unnoticed"


def test_r2_5_verify_notices_a_changed_work_file_and_candidate():
    import evidence

    with tempfile.TemporaryDirectory() as td:
        rdir = finalized(Path(td))
        (rdir / "raw" / "work" / "notes.md").write_text("my notez")  # same size
        assert evidence.verify(rdir), "a changed work file went unnoticed"
    with tempfile.TemporaryDirectory() as td:
        rdir = finalized(Path(td))
        c = next((rdir / "raw" / "candidates").glob("*.npy"))
        data = bytearray(c.read_bytes())
        data[-1] ^= 1
        c.write_bytes(bytes(data))
        assert evidence.verify(rdir), "a changed candidate went unnoticed"


@with_runs
def test_r2_6_text_divergence_does_not_block_recovery(root):
    rdir = score_fixture(root, "lock-a5-ep7")
    lock_refs(root)
    plain = C.REFERENCE_RUNS / "lock-a5-ep7-val-plain" / "generations.jsonl"
    rows = [json.loads(x) for x in plain.read_text().splitlines()]
    for r in rows:
        r["text"] = "different rounding gave another wording"
    plain.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fake, _ = fake_evalset([])
    run_score(rdir, fake)
    s = json.loads((C.RUNS / "harness-scratch-s-validation" / "summary.json").read_text())
    assert s["recovery"] == 0.0 and s["identical_text_to_reproducibility_run"] == 0.0, s


@with_runs
def test_r2_6_a_checkpoint_that_is_not_the_pinned_one_gets_no_recovery(root):
    rdir = score_fixture(root, "lock-a5-ep7")
    lock_refs(root)
    C.CHECKPOINTS = {"lock-a5-ep7": "0" * 64}
    fake, _ = fake_evalset([])
    run_score(rdir, fake)
    s = json.loads((C.RUNS / "harness-scratch-s-validation" / "summary.json").read_text())
    assert s["recovery"] is None and any("pinned" in e for e in s["reference_errors"]), s


def test_r2_7_a_halt_leaves_a_status_file():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rdir, cfg, stage = claude_driver(root, "", False)
        ev = json.dumps([])

        class Odd(A.ScriptedAgent):
            def collect(self, raw_dir, seq, now):
                return A.classify([], "Error: a message nobody has seen", 1, now)

        d = D.Driver(rdir, cfg, Odd([py("x")]), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=lambda s: None)
        rc = d.run()
        text = (rdir / "status.txt").read_text()
        assert rc == D.HALTED and "halted" in text and "a message nobody has seen" in text and "--clear-halt" in text


def test_r2_7_queue_reports_a_halt():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "code" / "harness").mkdir(parents=True)
        (root / "code" / ".venv" / "bin").mkdir(parents=True)
        shutil.copy(HARNESS / "queue.sh", root / "code" / "harness" / "queue.sh")
        d = root / "code" / "runs" / "agent-L-neutral-1"
        d.mkdir(parents=True)
        (d / "status.txt").write_text("state: halted\nreason: something odd\nresume, after looking at the reason: X --clear-halt\n")
        py_ = root / "code" / ".venv" / "bin" / "python"
        py_.write_text("#!/bin/sh\nexit 3\n")
        py_.chmod(0o755)
        r = subprocess.run(["zsh", str(root / "code" / "harness" / "queue.sh"), "100"], capture_output=True, text=True, env={"PATH": "/bin:/usr/bin"})
        assert r.returncode == 3
        text = (root / "code" / "runs" / "queue-status.txt").read_text()
        assert "HALTED" in text and "something odd" in text and "--clear-halt" in text and "L-neutral-1" in text


def test_r2_no_test_can_reach_the_real_runs_folder():
    real = (HARNESS.parent / "runs").resolve()
    assert C.RUNS.resolve() != real and real not in C.RUNS.resolve().parents
    try:
        shutil.rmtree(real / "nothing-here")
        raise AssertionError("not guarded")
    except RuntimeError as e:
        assert "real runs folder" in str(e)



# ---- after the pilot (L-neutral-1)

KILLED_AT_EXEC = """#!/bin/sh
# an interpreter macOS refuses: killed before it runs a line, as the pilot's runners were
n=$(cat "%(dir)s/tries" 2>/dev/null || echo 0)
echo $((n + 1)) > "%(dir)s/tries"
if [ "$n" -ge %(fail)s ]; then
  exec "%(python)s" "$@"
fi
echo "refused by the system" >&2
kill -9 $$
"""


def refusing_interpreter(root: Path, fail: int) -> Path:
    d = root / "interp"
    d.mkdir()
    (d / "python").write_text(KILLED_AT_EXEC % {"dir": d, "fail": fail, "python": sys.executable})
    (d / "python").chmod(0o755)
    return d


def launch_run(root, fail, limit, seconds=60):
    """The real ClaudeAgent and call_runner, with an interpreter that dies at launch `fail` times."""
    signal.alarm(seconds)  # the old driver never stops
    bindir = fake_cli(root, 0, py("x = 1"))
    interp = refusing_interpreter(root, fail)
    keep = sys.executable, os.environ["PATH"], getattr(C, "MAX_LAUNCH_FAILURES", None)
    rdir, cfg, stage = claude_driver(root, bindir, False)
    sys.executable = str(interp / "python")
    os.environ["PATH"] = f"{bindir}:{os.environ['PATH']}"
    C.MAX_LAUNCH_FAILURES = limit
    clock = {"t": time.time()}

    def sleep(s):  # backoff passes at once, but a runner gets real time to start
        clock["t"] += s
        time.sleep(0.05)

    try:
        d = D.Driver(rdir, cfg, A.ClaudeAgent(stage / "cli", "low"), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=sleep,
                     now=lambda: clock["t"])
        rc = d.run(max_turns=1)
    finally:
        sys.executable, os.environ["PATH"], C.MAX_LAUNCH_FAILURES = keep
    signal.alarm(0)
    return rc, rdir, interp


def test_r3_1_runners_that_die_at_launch_stop_the_run_with_the_reason():
    with tempfile.TemporaryDirectory() as td:
        rc, rdir, interp = launch_run(Path(td).resolve(), fail=10 ** 6, limit=4)
        st = json.loads((rdir / "state.json").read_text())
        assert rc == D.HALTED and "failed to start 4 times" in st["halted"] and "-9" in st["halted"], (rc, st["halted"])
        assert st["inflight"] is None and st["turn"] == 0
        assert (interp / "tries").read_text().strip() == "4"
        rows = [json.loads(x) for x in (rdir / "launch_failures.jsonl").read_text().splitlines()]
        assert len(rows) == 4 and all(r["runner_exit"] == -9 for r in rows) and "refused by the system" in rows[0]["runner_log"]


def test_r3_1_a_failed_launch_leaves_no_bookkeeping_files():
    with tempfile.TemporaryDirectory() as td:
        rc, rdir, _ = launch_run(Path(td).resolve(), fail=10 ** 6, limit=5)
        left = [p.name for p in (rdir / "turns" / "t001" / "raw").iterdir()]
        assert not left, left[:8]


def test_r3_1_a_short_outage_recovers_by_itself_with_growing_waits():
    with tempfile.TemporaryDirectory() as td:
        rc, rdir, _ = launch_run(Path(td).resolve(), fail=3, limit=20)
        st = json.loads((rdir / "state.json").read_text())
        assert st["turn"] == 1 and st["launch_failures"] == 0 and st["launch_failures_total"] == 3, st
        assert st["waited_s"] >= 1 + 2 + 4 - 0.5
        log = (rdir / "driver.log").read_text()
        assert "waiting 4 s" in log


def test_r3_1_a_late_runner_cannot_start_after_its_attempt_is_discarded():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        raw = root / "raw"
        raw.mkdir()
        (raw / "call7.lock").touch()
        (raw / "call7.prompt").write_text("p")
        (raw / "call7.cmd.json").write_text(json.dumps([str(root / "marker.sh")]))
        (root / "marker.sh").write_text(f"#!/bin/sh\ntouch {root}/ran\n")
        (root / "marker.sh").chmod(0o755)
        fd = os.open(raw / "call7.lock", os.O_RDWR)  # the driver, holding the lock while it gives the attempt up
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)
        code = ("import sys, time, os\nsys.argv = ['r', %r, '7', %r, '5']\nimport runpy, fcntl\nreal = fcntl.flock\n"
                "def wait(f, how):\n    real(f, fcntl.LOCK_EX)\nfcntl.flock = wait\n"
                "runpy.run_path(%r, run_name='__main__')" % (str(raw), str(root), str(HARNESS / "call_runner.py")))
        p = subprocess.Popen([sys.executable, "-c", code])  # it has opened the lock and waits for it
        time.sleep(0.5)
        A.ClaudeAgent(root, "low").discard_attempt(raw, 7)
        os.close(fd)
        p.wait(timeout=10)
        assert not (root / "ran").exists(), "a runner launched an attempt that had been given up"


def test_r3_2_blocks_keep_their_source_in_the_work_area():
    import kernel_main as K2

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        K2.BLOCKS = str(root / "blocks")
        g = {}
        code = "def twice(x):\n    return x * 2\n\ntwice(4)"
        name = K2.keep_source(100, code)
        r = K2.captured_run(code, g, 5, str(root / "c.txt"), name)
        assert r["output"] == "8\n" and (root / "blocks" / "block_100.py").read_text() == code
        r = K2.captured_run("import inspect\nprint(inspect.getsource(twice))", g, 5, str(root / "c.txt"), K2.keep_source(101, "x"))
        assert "return x * 2" in r["output"], r  # the source of a function from an earlier block, as python100 needed
        r = K2.captured_run("1 / 0", g, 5, str(root / "c.txt"), K2.keep_source(102, "1 / 0"))
        assert "block_102.py" in r["output"] and "ZeroDivisionError" in r["output"]


def handoff_driver(root, turns):
    rdir, cfg, stage = claude_driver(root, "", False)
    d = D.Driver(rdir, cfg, A.ScriptedAgent([]), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=lambda s: None)
    d.st = D.new_state(cfg, "TASK")
    for t in range(1, turns + 1):
        (d.tdir(t) / "answer.md").write_text(f"Explanation number {t} says why.\n\n```python\nsecret_code_{t} = 1\n```")
    d.st["turn"] = turns
    return d


def test_r3_2_a_handoff_carries_the_recent_explanations():
    with tempfile.TemporaryDirectory() as td:
        d = handoff_driver(Path(td), 12)
        d.new_conversation("LAST RESULT")
        text = d.st["next"]
        assert "Explanation number 12 says why." in text and "Explanation number 5 says why." in text, text[-400:]
        assert "Explanation number 4 says why." not in text, "older replies stay out: the handoff is compact"
        assert "secret_code_" not in text, "the code stays in /work/blocks, not in the handoff"
        assert text.index("number 5") < text.index("number 12") < text.index("LAST RESULT")


def test_r3_2_the_handoff_is_generic_across_models_and_framings():
    import prompt as P

    src = (HARNESS / "prompt.py").read_text() + (C.TASK_DIR / "task.txt").read_text()
    assert "/work/blocks" in src and "current plan and the lines you have closed" in src
    a = P.handoff("T1", 40, 3, 50, 9, [], [], "n", "l", True, [(40, "x")])
    b = P.handoff("T2", 40, 3, 50, 9, [], [], "n", "l", True, [(40, "x")])
    assert a.replace("T1", "") == b.replace("T2", "")
    for word in ("lock", "password", "hiding", "capability", "loop", "weak"):
        assert not re.search(rf"\b{word}", a.lower()), word


def test_r3_3_a_short_npy_cannot_declare_a_huge_header():
    import contextlib
    import host_edits as H
    import safeio

    with tempfile.TemporaryDirectory() as td:
        stage = make_stage(Path(td))
        d = stage / "work" / "edited"
        d.mkdir()
        p = d / f"{DOWN}.npy"
        p.write_bytes(b"\x93NUMPY\x02\x00" + (0xffffffff).to_bytes(4, "little"))
        assert p.stat().st_size == 12
        m = edit_store.Model(stage / "model" / "model.safetensors")
        asked = []

        class Recording:
            def __init__(self, f):
                self.f = f

            def read(self, n=-1):
                asked.append(n)
                return self.f.read(min(n, 1 << 20)) if n >= 0 else self.f.read()

            def __getattr__(self, k):
                return getattr(self.f, k)

        real = safeio.open_read

        @contextlib.contextmanager
        def recording(*a, **k):
            with real(*a, **k) as f:
                yield Recording(f)

        safeio.open_read = recording
        try:
            t, s, why = H.read_edits(m, stage / "work")
        finally:
            safeio.open_read = real
        assert not t and why, why
        assert max(asked) <= H.NPY_HEADER_MAX if hasattr(H, "NPY_HEADER_MAX") else max(asked) < 1 << 20, f"asked to read {max(asked)} bytes"


def test_r3_4_verify_notices_a_lost_candidate_index_and_store():
    import evidence

    with tempfile.TemporaryDirectory() as td:
        rdir = finalized(Path(td))
        assert not evidence.verify(rdir)
        (rdir / "candidates.jsonl").unlink()
        shutil.rmtree(rdir / "raw" / "candidates")
        assert evidence.verify(rdir), "losing the whole candidate record went unnoticed"


def test_r3_4_verify_notices_an_edited_candidate_index():
    import evidence

    with tempfile.TemporaryDirectory() as td:
        rdir = finalized(Path(td))
        rows = (rdir / "candidates.jsonl").read_text().splitlines()
        (rdir / "candidates.jsonl").write_text(rows[0].replace('"at": "end"', '"at": "elsewhere"') + "\n" + "\n".join(rows[1:]))
        assert evidence.verify(rdir), "a changed index went unnoticed"
        ev = json.loads((rdir / "evidence.json").read_text())
        assert ev["candidate_index"]["sha256"]


def test_r3_4_a_run_from_before_the_index_hash_is_still_checked():
    import evidence

    with tempfile.TemporaryDirectory() as td:
        rdir = finalized(Path(td))
        ev = json.loads((rdir / "evidence.json").read_text())
        ev.pop("candidate_index", None)  # as agent-L-neutral-1's was written
        (rdir / "evidence.json").write_text(json.dumps(ev))
        assert not evidence.verify(rdir)
        (rdir / "candidates.jsonl").unlink()
        assert evidence.verify(rdir), "the old format's counts must still catch a lost index"


def test_r3_4_check_changes_nothing():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rdir = finalized(root)
        stage = Path(json.loads((rdir / "run_config.json").read_text())["stage"])
        name = rdir.name
        keep = C.RUNS, C.STAGE_ROOT
        import cleanup

        before = sorted((str(p), p.stat().st_mtime_ns) for p in rdir.rglob("*"))
        (rdir / "candidates.jsonl").unlink()
        saved = sys.argv
        C.run_dir_real = C.run_dir
        C.run_dir = lambda n: rdir
        sys.argv = ["cleanup.py", "--run", "x", "--check"]
        try:
            cleanup.main()
            raise AssertionError("a lost index passed --check")
        except SystemExit as e:
            assert "nothing was changed" in str(e)
        finally:
            sys.argv = saved
            C.run_dir = C.run_dir_real
        assert stage.exists() and rdir.exists()
        now = {p[0] for p in before} - {str(p) for p in rdir.rglob("*")}
        assert now == {str(rdir / "candidates.jsonl")}, now


def test_r3_5_stale_requests_are_captured_before_readiness():
    """A block sent the moment the kernel says it is ready must run, not be taken for one from before the boot."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        for d in ("ctl", "inbox", "kernel"):
            (root / d).mkdir()
        shutil.copy(C.KERNEL_SRC / "kernel_main.py", root / "kernel" / "kernel_main.py")
        (root / "kernel" / "prelude.py").write_text("x = 1\n")
        code = ("import sys, os\nsys.path.insert(0, %r)\nimport kernel_main as K\nreal = K.write_atomic\n"
                "def ready_then_block(path, text):\n    real(path, text)\n"
                "    if path.endswith('ready.json'):\n        open(os.path.join(K.INBOX, 'req_1.py'), 'w').write('print(40 + 2)')\n"
                "K.write_atomic = ready_then_block\nK.main()" % str(root / "kernel"))
        env = {**os.environ, "KERNEL_CTL": str(root / "ctl"), "KERNEL_INBOX": str(root / "inbox"), "KERNEL_BLOCKS": str(root / "blocks")}
        p = subprocess.Popen([sys.executable, "-c", code], env=env)
        try:
            t0 = time.time()
            while not (root / "ctl" / "out_1.json").exists():
                assert time.time() - t0 < 8, "the block sent at readiness never ran"
                time.sleep(0.05)
            assert "42" in json.loads((root / "ctl" / "out_1.json").read_text())["output"]
        finally:
            p.kill()


def flags(code):
    import astcheck

    return astcheck.check_block(code)[1]


def test_r3_6_reading_the_edit_folder_is_not_writing_it():
    import audit
    import astcheck

    def flagged(code):
        return sorted(set(astcheck.check_block(code)[1]) | set(audit.hits(code)))

    for code in ("for f in os.listdir('/work/edited'):\n    shutil.copy('/work/edited/' + f, '/work/best_1/' + f)",
                 "shutil.copytree('/work/edited', '/work/best_2')", "print(os.listdir('/work/edited'))",
                 "x = np.load('/work/edited/a.npy')", "for fn in os.listdir('/work/edited'):\n    open('/work/c/' + fn, 'wb').write(open('/work/edited/' + fn, 'rb').read())",
                 "# never write to /work/edited by hand\nx = 1"):
        assert not [f for f in flagged(code) if "edit folder" in f], code
    for code in ("np.save('/work/edited/model.layers.0.mlp.down_proj.weight.npy', w)", "open('/work/edited/x.npy', 'wb').write(b)",
                 "shutil.copy('/work/best_1/x.npy', '/work/edited/x.npy')", "os.remove('/work/edited/x.npy')", "d = '/work/edited'\nnp.save(d + '/x.npy', w)",
                 "from pathlib import Path\nPath('/work/edited/x.npy').write_bytes(b)", "p = '/work/edited/x.npy'\nwith open(p, 'wb') as f:\n    f.write(b)",
                 "torch.save(w, f'/work/edited/{name}.npy')", "os.system('cp /work/best/x.npy /work/edited/')", "shutil.move('/work/edited/x.npy', '/work/old/')",
                 "shutil.rmtree('/work/edited')", "for p in glob.glob('/work/edited/*'):\n    os.remove(p)", "np.save(os.path.join('/work', 'edited', 'x.npy'), w)"):
        assert [f for f in flagged(code) if "edit folder" in f], code


def test_r3_6_names_are_flagged_when_called_not_when_mentioned():
    import astcheck

    def f(code):
        return astcheck.check_block(code)[1]

    for code in ("import re\npat = re.compile('x')", "model.eval()", "solve = 3\nprint(solve)", "plan = {'solve': 1}", "from re import compile\nc = compile('x')",
                 "# np.linalg.solve(a, b) would be forbidden\nx = 1", "text = 'eval(x)'"):
        assert not f(code), (code, f(code))
    for code, want in (("x = np.linalg.solve(a, b)", "solve"), ("solve(a, b)", "solve"), ("from scipy.linalg import solve\nsolve(a, b)", "solve"),
                       ("torch.linalg.solve(a, b)", "solve"), ("s = np.linalg.solve\ns(a, b)", "solve"), ("np.linalg.lstsq(a, b)", "lstsq"), ("np.linalg.pinv(a)", "pinv"),
                       ("torch.linalg.pinv(a)", "pinv"), ("compile('1', 'f', 'eval')", "compile"), ("eval('1')", "eval"), ("exec('x = 1')", "exec"),
                       ("g = eval\ng('1')", "eval"), ("import builtins\nbuiltins.eval('1')", "builtins")):
        assert any(want in x for x in f(code)), (code, f(code))


# ---- third group

def finalize_run(root, fail, limit=None, reply=None):
    """A real ForwardTool and finalize.py, behind an interpreter killed at launch `fail` times."""
    signal.alarm(120)
    stage, rdir = pass_fixture(root)
    (rdir / "first_prompt.txt").write_text("TASK")
    (rdir / "task.txt").write_text("TASK")
    cfg = json.loads((rdir / "run_config.json").read_text())
    interp = refusing_interpreter(root, fail)
    agent = A.ScriptedAgent([reply or FINAL_R, FINAL_R, FINAL_R, FINAL_R, FINAL_R])
    clock = {"t": 1_000.0}

    def sleep(s):
        clock["t"] += s

    keep = sys.executable, getattr(C, "MAX_FINALIZE_FAILURES", None)
    sys.executable = str(interp / "python")
    C.MAX_FINALIZE_FAILURES = limit or 8
    try:
        d = D.Driver(rdir, cfg, agent, StubKernel(), D.ForwardTool(rdir), lambda t: (t, 0), sleep=sleep, now=lambda: clock["t"])
        d.st = D.new_state(cfg, "TASK")
        d.st["passes"] = d.st["budget"]
        d.st["kernel_incarnation"] = "boot"
        d.save()
        rc = d.run(max_turns=5)
    finally:
        sys.executable, C.MAX_FINALIZE_FAILURES = keep
    signal.alarm(0)
    return rc, rdir, agent, stage, clock


def test_r4_1_a_finalize_that_fails_says_nothing_to_the_agent():
    with tempfile.TemporaryDirectory() as td:
        rc, rdir, agent, stage, clock = finalize_run(Path(td).resolve(), fail=3)
        told = "\n".join(agent.calls)
        assert "could not be submitted" not in told and "Fix them" not in told, told[:300]
        st = json.loads((rdir / "state.json").read_text())
        assert st["phase"] != "work", st["phase"]  # it went on to the debrief, with no second answer
        assert (rdir / "submitted" / "summary.json").exists()
        assert st["waited_s"] >= 30 + 60 + 120 - 1, st["waited_s"]  # its own wait: pending runs before wait_until


def test_r4_1_a_finalize_that_keeps_failing_halts_with_the_edit_frozen():
    with tempfile.TemporaryDirectory() as td:
        rc, rdir, agent, stage, clock = finalize_run(Path(td).resolve(), fail=10 ** 6, limit=3)
        st = json.loads((rdir / "state.json").read_text())
        assert rc == D.HALTED and "finalize failed 3 times" in st["halted"], (rc, st["halted"])
        assert st["pending"]["kind"] == "finalize", "it must stay pending, to go on after --clear-halt"
        assert len(agent.calls) == 1, "the agent must not get another turn"
        assert "finalize failed" in (rdir / "status.txt").read_text()


def test_r4_1_a_rule_refusal_still_reaches_the_agent():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        stage, rdir = pass_fixture(root)
        d = stage / "work" / "edited"
        d.mkdir()
        m = edit_store.Model(stage / "model" / "model.safetensors")
        np.save(d / f"{DOWN}.npy", m.f32(DOWN) * 2)  # a whole-tensor rescale, written past save_edit
        (rdir / "first_prompt.txt").write_text("TASK")
        (rdir / "task.txt").write_text("TASK")
        cfg = json.loads((rdir / "run_config.json").read_text())
        agent = A.ScriptedAgent([FINAL_R, FINAL_R, FINAL_R])
        drv = D.Driver(rdir, cfg, agent, StubKernel(), D.ForwardTool(rdir), lambda t: (t, 0), sleep=lambda s: None)
        drv.st = D.new_state(cfg, "TASK")
        drv.st["passes"] = drv.st["budget"]
        drv.st["kernel_incarnation"] = "boot"
        drv.save()
        drv.run(max_turns=2)
        assert "could not be submitted" in agent.calls[1] and "rescales" in agent.calls[1], agent.calls[1:]


def done_run_without_stage(name="donerun"):
    rdir = C.run_dir(name)
    rdir.mkdir(parents=True)
    (rdir / "run_config.json").write_text(json.dumps({"run": name, "stage": str(rdir / "gone"), "effort": "low", "mem_gb": 14}))
    (rdir / "state.json").write_text(json.dumps({"done": True, "turn": 266}))
    (rdir / "driver.log").write_text("the end\n")
    (rdir / "status.txt").write_text("state: done\n")
    return rdir


def test_r4_2_a_finished_run_needs_no_stage_and_its_logs_stay():
    rdir = done_run_without_stage()
    try:
        r = subprocess.run([sys.executable, "-I", str(HARNESS / "driver.py"), "--run", "donerun", "--max-wall-seconds", "60"],
                           capture_output=True, text=True, env={**os.environ, "HARNESS_RUNS": str(C.RUNS)})
        assert r.returncode == 0, r.stderr[-300:]
        assert (rdir / "driver.log").read_text() == "the end\n" and (rdir / "status.txt").read_text() == "state: done\n"
    finally:
        shutil.rmtree(rdir)


def test_r4_3_launch_retries_last_two_hours():
    assert C.MAX_LAUNCH_FAILURES == 125
    total = sum(min(2.0 ** (n - 1), C.LAUNCH_BACKOFF_MAX) for n in range(1, C.MAX_LAUNCH_FAILURES))
    assert 1.9 * 3600 < total < 2.1 * 3600, total
    assert "125 in a row" in (HARNESS / "README.md").read_text()


def queue_tree(root: Path, script: str) -> Path:
    (root / "code" / "harness").mkdir(parents=True)
    (root / "code" / ".venv" / "bin").mkdir(parents=True)
    shutil.copy(HARNESS / "queue.sh", root / "code" / "harness" / "queue.sh")
    py_ = root / "code" / ".venv" / "bin" / "python"
    py_.write_text(script)
    py_.chmod(0o755)
    return root / "code" / "harness" / "queue.sh"


DIES_THEN = """#!/bin/sh
n=$(cat "%(c)s" 2>/dev/null || echo 0)
echo $((n + 1)) > "%(c)s"
[ "$n" -lt %(n)s ] && kill -9 $$
exit 0
"""


def test_r4_4_a_driver_killed_at_exec_is_retried():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        q = queue_tree(root, DIES_THEN % {"c": root / "count", "n": 3})
        r = subprocess.run(["zsh", str(q), "100"], capture_output=True, text=True,
                           env={"PATH": "/bin:/usr/bin", "QUEUE_KILL_WAIT": "0"})
        assert r.returncode == 0 and (root / "count").read_text().strip() == "4", (r.returncode, r.stdout[-200:])


def test_r4_4_a_driver_that_keeps_dying_stops_the_queue():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        q = queue_tree(root, DIES_THEN % {"c": root / "count", "n": 10 ** 6})
        r = subprocess.run(["zsh", str(q), "100"], capture_output=True, text=True,
                           env={"PATH": "/bin:/usr/bin", "QUEUE_KILL_WAIT": "0", "QUEUE_KILL_RETRIES": "3"})
        assert r.returncode == 137 and (root / "count").read_text().strip() == "4"
        assert "killed" in (root / "code" / "runs" / "queue-status.txt").read_text()


def test_r4_5_a_cli_killed_at_exec_backs_off():
    signal.alarm(60)
    keep = os.environ["PATH"], C.MAX_OUTAGES
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        d = root / "bin"
        d.mkdir()
        (d / "claude").write_text("#!/bin/sh\nkill -9 $$\n")
        (d / "claude").chmod(0o755)
        rdir, cfg, stage = claude_driver(root, str(d), False)
        os.environ["PATH"] = f"{d}:{os.environ['PATH']}"
        clock = {"t": time.time()}

        def sleep(s):
            clock["t"] += s
            time.sleep(0.02)

        try:
            drv = D.Driver(rdir, cfg, A.ClaudeAgent(stage / "cli", "low"), StubKernel(), StubForward(rdir), lambda t: (t, 0), sleep=sleep,
                           now=lambda: clock["t"])
            drv.st = D.new_state(cfg, "TASK")
            drv.st["kernel_incarnation"] = "boot"
            drv.st["session_new"] = False
            drv.st["turn"] = 3
            (drv.tdir(3) / "answer.md").write_text("x")
            drv.save()
            rc = drv.run()
        finally:
            os.environ["PATH"], C.MAX_OUTAGES = keep
        st = json.loads((rdir / "state.json").read_text())
        assert rc == D.HALTED and "killed" in st["halted"]
        assert st["waited_s"] >= 30 + 60 + 120 + 240 + 480 - 1, st["waited_s"]
        assert st["handoffs"] <= 1, st["handoffs"]
    signal.alarm(0)


def test_r4_6_flags_that_need_a_tight_call_still_fire():
    import audit

    for code, want in (("best = argmin(losses)", "(b)"), ("k = max(score)", "(b)"), ("idx = sorted(loss_of)", "(b)"),
                       ("g = X.T @ X + lam * eye", "normal equations"), ("x = 1  # argmin(loss) would be forbidden\ny = 2", None)):
        got = audit.hits(code)
        assert (any(want in g for g in got) if want else not got), (code, got)


def test_r4_7_solve_imported_from_linalg_is_flagged():
    for code in ("from numpy.linalg import solve as s\ns(a, b)", "from scipy.linalg import solve\nmap(solve, xs)",
                 "from torch.linalg import solve as t\nx = list(map(t, xs))"):
        assert any("solve" in f for f in flags(code)), code
    assert not flags("from mylib import solve\nmap(solve, xs)"), "a solve of the agent's own is not a linear solver"


def test_r4_8_tracked_path_text_is_capped():
    import ast
    import astcheck
    import tracemalloc

    lines = ["a = '/work/edited/' + 'x' * 4000"] + [f"a = a + a" for _ in range(22)] + ["np.save(a, w)"]
    code = "\n".join(lines)
    tracemalloc.start()
    flagged = astcheck.check_block(code)[1]
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 64 * 2 ** 20, f"{peak / 2 ** 20:.0f} MB"
    assert any("edit folder" in f for f in flagged)
    env = astcheck._path_names(ast.parse(code))
    assert max(len(v) for v in env.values()) <= 4096


def test_r4_9_a_malformed_header_or_huge_block_kills_nothing():
    import astcheck
    import host_edits as H

    with tempfile.TemporaryDirectory() as td:
        stage = make_stage(Path(td))
        d = stage / "work" / "edited"
        d.mkdir()
        p = d / f"{DOWN}.npy"
        body = b"{'descr': '<f4', 'fortran_order': False, 'shape': (8, 32"  # the tuple never closes
        p.write_bytes(b"\x93NUMPY\x02\x00" + len(body).to_bytes(4, "little") + body)
        m = edit_store.Model(stage / "model" / "model.safetensors")
        t, _, why = H.read_edits(m, stage / "work")
        assert not t and why, why
    assert astcheck.check_block("x = " + "(" * 100_000 + "1" + ")" * 100_000) == ([], [])
    assert astcheck.check_block("x = " + "-" * 200_000 + "1") == ([], [])
    assert astcheck.check_block("x = " + "+".join(["1"] * 300_000)) is not None


def test_r4_10_an_unreadable_started_file_means_not_started():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        raw = root / "raw"
        raw.mkdir()
        (raw / "call1.lock").touch()
        (raw / "call1.started").write_text('{"pid": 1, "sta')  # the runner died in the middle of writing it
        agent = A.ClaudeAgent(root, "low")
        assert agent.await_attempt(raw, 1, lambda s: None, float("inf")) == "not_launched"
        holder = subprocess.Popen([sys.executable, "-c", f"import fcntl,os,time\nfd=os.open({str(raw / 'call1.lock')!r},os.O_RDWR)\n"
                                   "fcntl.flock(fd,fcntl.LOCK_EX)\nprint('held',flush=True)\ntime.sleep(30)"], stdout=subprocess.PIPE, text=True)
        holder.stdout.readline()
        try:
            t = {"n": 0.0}
            assert agent.await_attempt(raw, 1, lambda s: t.update(n=t["n"] + 1), 3.0, lambda: t["n"]) == "running"
        finally:
            holder.kill()


RUN_ALL_QUEUE = """#!/bin/zsh
n=$(cat "%(root)s/calls" 2>/dev/null || echo 0)
n=$((n + 1))
echo $n > "%(root)s/calls"
echo "$QUEUE_REVIEWED $1" >> "%(root)s/args"
sleep %(sleep)s
codes=(%(codes)s)
exit ${codes[$n]}
"""


def run_all_tree(root: Path, codes: str, sleep: str = "0") -> Path:
    h = root / "code" / "harness"
    h.mkdir(parents=True)
    shutil.copy(HARNESS / "run_all.sh", h / "run_all.sh")
    (h / "queue.sh").write_text(RUN_ALL_QUEUE % {"root": root, "codes": codes, "sleep": sleep})
    b = root / "bin"
    b.mkdir()
    (b / "caffeinate").write_text(f'#!/bin/sh\necho "$1" >> "{root}/awake"\nshift\nexec "$@"\n')
    (b / "caffeinate").chmod(0o755)
    return h / "run_all.sh"


def call_run_all(root, script, *args, pause="0"):
    return subprocess.run(["zsh", str(script), *args], capture_output=True, text=True,
                          env={"PATH": f"{root}/bin:/bin:/usr/bin", "RUN_ALL_PAUSE": pause})


def test_the_resume_line_keeps_the_run_list_and_only_a_real_review():
    for env, want in (({"QUEUE_REVIEWED": "0"}, "go on with: code/harness/queue.sh 100"),
                      ({"QUEUE_REVIEWED": "1"}, "go on with: QUEUE_REVIEWED=1 code/harness/queue.sh 100"),
                      ({"QUEUE_REVIEWED": "1", "QUEUE_RUNS": "a b"},
                       "go on with: QUEUE_RUNS='a b' QUEUE_REVIEWED=1 code/harness/queue.sh 100")):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            q = queue_tree(root, "#!/bin/sh\nexit 2\n")
            subprocess.run(["zsh", str(q), "100"], capture_output=True, text=True, env={"PATH": "/bin:/usr/bin", **env})
            assert want in (root / "code" / "runs" / "queue-status.txt").read_text().splitlines()


def test_two_run_alls_from_two_folders_share_a_relative_lock():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        script = run_all_tree(root, "0", sleep="3")
        (root / "a").mkdir()
        (root / "b").mkdir()
        env = {"PATH": f"{root}/bin:/bin:/usr/bin", "HARNESS_RUNS": "review-output"}
        first = subprocess.Popen(["zsh", str(script), "5"], cwd=root / "a", env=env, stdout=subprocess.DEVNULL)
        lock = root / "review-output" / ".run_all.lock"
        deadline = time.time() + 20
        while not (root / "calls").exists() and time.time() < deadline:
            time.sleep(0.05)
        second = subprocess.run(["zsh", str(script), "5"], cwd=root / "b", env=env, capture_output=True, text=True)
        assert second.returncode == 4, (second.returncode, second.stdout[-200:])
        assert first.wait(timeout=30) == 0 and lock.exists() and not (root / "b" / "review-output").exists()


def test_r4_11_run_all_keeps_going_until_done():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        r = call_run_all(root, run_all_tree(root, "2 2 0"), "500")
        assert r.returncode == 0, (r.returncode, r.stdout[-200:], r.stderr[-200:])
        assert (root / "calls").read_text().strip() == "3"
        assert (root / "awake").read_text().strip() == "-i", "it must keep the machine awake"
        assert (root / "args").read_text().splitlines() == ["1 500"] * 3  # QUEUE_REVIEWED=1 and the segment


def test_r4_11_run_all_stops_on_a_halt():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        r = call_run_all(root, run_all_tree(root, "2 3 0"), "500")
        assert r.returncode == 3 and (root / "calls").read_text().strip() == "2", (r.returncode, r.stdout[-200:])


def test_r4_11_run_all_exits_2_at_its_total_limit():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        r = call_run_all(root, run_all_tree(root, "2 2 2 2 2 2 2 2", sleep="1"), "500", "2")
        assert r.returncode == 2 and "call again" in r.stdout, (r.returncode, r.stdout[-200:])
        lines = (root / "args").read_text().splitlines()
        assert len(lines) <= 3 and int(lines[0].split()[1]) <= 2, lines  # no segment longer than what is left


def test_r4_11_a_quick_exit_2_does_not_spin():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        t0 = time.time()
        r = call_run_all(root, run_all_tree(root, "2 0"), "500", pause="2")
        assert r.returncode == 0 and time.time() - t0 >= 2, "waiting for the model must not become a tight loop"


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    only = [a for a in sys.argv[1:] if a.startswith("test_")]
    if "--list-failures" in sys.argv:  # one process per test: against old code a test can hang or leave state behind
        for f in TESTS:
            try:
                r = subprocess.run([sys.executable, "-I", __file__, f.__name__], capture_output=True, text=True, timeout=90)
                last = (r.stderr.strip().splitlines() or [""])[-1]
                print(f"passes {f.__name__}" if r.returncode == 0 else f"FAILS {f.__name__}: {last[:130]}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"FAILS {f.__name__}: hung", flush=True)
    else:
        for f in TESTS:
            if only and f.__name__ not in only:
                continue
            f()
            print("ok", f.__name__, flush=True)
