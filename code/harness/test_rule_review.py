"""Tests of the review before execution (CPU, no Docker, no model calls): the reviewer is canned, or fake_claude.py behind the real
ClaudeAgent and call_runner. They test the gate, not the reviewer's judgement: that is rule_review_replay.py.

Usage: code/.venv/bin/python -I code/harness/test_rule_review.py [test names]
"""

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
import testenv  # noqa: E402,F401  (first: the runs folder moves to a temp folder)
import config as C  # noqa: E402

if not (C.MODELS / "lock-a5-ep7" / "config.json").exists():  # a worktree has no models, and test_harness reads their configs at import
    C.MODELS = Path(tempfile.mkdtemp(prefix="review-test-models-"))
    for tag in ("lock-a5-ep7", "piweak-a2", "honest-a5-ep7"):
        (C.MODELS / tag).mkdir()
        (C.MODELS / tag / "config.json").write_text("{}")
import test_harness as T  # noqa: E402
import agent as A  # noqa: E402
import audit  # noqa: E402
import driver as D  # noqa: E402
import prompt as P  # noqa: E402
import redact  # noqa: E402
import rule_review as R  # noqa: E402

REVIEW_ID = re.compile(r"^review_id: ([0-9a-f]{64})$", re.M)
SECRET = "kept-from-the-reviewer"


class Gated(R.ReviewMixin, D.Driver):
    pass


def reply(decision="allow", grade="A", rules=(), reason="decided by statement one", **over):
    def make(prompt):
        return json.dumps({"review_id": REVIEW_ID.search(prompt).group(1), "decision": decision, "grade": grade,
                           "rules": list(rules), "reason": reason, **over})
    return make


REJECT = reply("reject", "C", [4], "the planned write on each token")
UNSURE = reply("uncertain", "none")


def events(session, text, model=C.MODEL_ID, tools=()):
    ev = A.scripted_events(session, text)
    ev[0]["model"] = model
    ev[0]["tools"] = list(tools)
    return ev


class Reviewer:
    """Plays the CLI's side of RuleGate, one entry per attempt: a reply maker, or a dict with the failure events."""

    def __init__(self, script, effort=R.EFFORT):
        self.script, self.effort = list(script), effort
        self.calls, self.sessions, self.never_started = [], [], set()

    def spawn(self, prompt_file, session, new, raw_dir, seq):
        assert new
        prompt = Path(prompt_file).read_text()
        self.calls.append(prompt)
        self.sessions.append(session)
        step = self.script.pop(0)
        if step == "no_launch":
            self.never_started.add((str(raw_dir), seq))
            return
        if step == "killed":
            return
        if isinstance(step, dict):
            ev = step["events"](session)
            (Path(raw_dir) / f"call{seq}.done").write_text(json.dumps(step.get("rc", 1)))
        else:
            ev = events(session, step(prompt))
        (Path(raw_dir) / f"call{seq}.json").write_text(json.dumps(ev))

    def await_attempt(self, raw_dir, seq, sleep=None, until=float("inf"), now=None):
        return "not_launched" if (str(raw_dir), seq) in self.never_started else "finished"

    def discard_attempt(self, raw_dir, seq):
        return {"runner_exit": None, "runner_log": ""}

    def collect(self, raw_dir, seq, now):
        p = Path(raw_dir) / f"call{seq}.json"
        if not p.exists():
            return A.classify([], "", -9, now)
        rc = Path(f"{p.with_suffix('')}.done")
        return A.classify(A.parse_events(p.read_text()), "", json.loads(rc.read_text()) if rc.exists() else 0, now)


def rate_limited(reset=500):
    def ev(session):
        e = A.scripted_events(session, "limit")
        e[-1].update(is_error=True, api_error_status=429, terminal_reason="api_error")
        e.insert(2, {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": 1_000_000 + reset}})
        return e
    return {"events": ev}


def failing(**fields):
    def ev(session):
        e = A.scripted_events(session, "")
        e[-1].update(is_error=True, **fields)
        return e
    return {"events": ev}


def world(root, replies, reviews, sanitize=None, crash=None, clock=None):
    rdir, cfg, agent, kernel, forward = T.setup(Path(root), 1, replies)
    cfg.update(rule_review=True, model_tag="lock-a5-ep7", model_dir="/x/organisms/models/lock-a5-ep7", framing="auditor", run="L-auditor-3")
    rev = Reviewer(reviews)
    clock = clock or T.Clock()

    def build(crash_=crash, rv=rev):
        return Gated(rdir, cfg, agent, kernel, forward, sanitize or (lambda t: (t, 0)), sleep=clock.sleep, now=clock.now, crash=crash_,
                     review_agent=rv)
    return rdir, build, agent, kernel, rev, clock


def state(rdir):
    return json.loads((rdir / "state.json").read_text())


def decisions(rdir):
    return [json.loads(p.read_text()) for p in sorted((rdir / "reviews").glob("*/decision.json"))]


def calls(rdir):
    return [json.loads(x) for x in (rdir / "tool_calls.jsonl").read_text().splitlines()] if (rdir / "tool_calls.jsonl").exists() else []


GEN = T.fwd(T.gen([0]))
PY = T.py("x = 1")
END = [GEN, T.FINAL, T.DEBRIEF]


def test_cleared_block_runs_and_is_bound_to_its_code():
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [reply()])
        assert build().run() == D.DONE
        assert kernel.executed == {1: 1} and len(rev.calls) == 1
        d = decisions(rdir)[0]
        assert d["decision"] == "allow" and d["code_sha256"] == R.sha("x = 1") and d["seq"] == 1 and d["session"] == rev.sessions[0]
        assert R.summary(rdir)["ran_without_approval"] == []


def test_every_outcome_but_allow_blocks_with_one_message():
    """The agent can't tell reject from a bad reply, and hears no reason, rule, grade or fix."""
    wrong = [REJECT, UNSURE, lambda p: "allow", lambda p: "```json\n" + reply()(p) + "\n```", lambda p: reply()(p) + " ok",
             lambda p: reply()(p).replace(REVIEW_ID.search(p).group(1), "0" * 64),
             reply(grade="B"), reply("allow", "A", [4]), reply("reject", "A", [4]), reply("reject", "none", []), reply("reject", "C", [5]),
             reply("uncertain", "A"), reply(extra=1), reply("ALLOW"), reply(rules=[True], grade="none", decision="reject"),
             lambda p: reply()(p)[:-1] + ', "decision": "allow"}', reply(reason="x" * (R.MAX_REASON_CHARS + 1)),
             {"events": lambda s: events(s, reply()("review_id: " + "a" * 64)), "rc": 0},
             failing(stop_reason="refusal")]
    for script in wrong:
        with tempfile.TemporaryDirectory() as td:
            rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [script])
            assert build().run() == D.DONE
            assert kernel.executed == {}, script
            d = decisions(rdir)[0]
            assert d["decision"] != "allow"
            assert R.BLOCK_MESSAGE in agent.calls[1]
            assert not re.search(r"planned write|rule 4|grade|uncertain|reject", agent.calls[1].split("OUTPUT")[0], re.I)
            assert state(rdir)["runs"] == 0 and state(rdir)["review_blocks"] == 1


def test_not_tool_free_wrong_model_or_stale_session_cannot_clear():
    cases = [lambda session, text: events(session, text, tools=["Read"]), lambda session, text: events(session, text, model="claude-haiku-5-5"),
             lambda session, text: events("other-session", text)]
    for make in cases:
        with tempfile.TemporaryDirectory() as td:
            rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [])
            rev.spawn = lambda prompt_file, session, new, raw_dir, seq, m=make: (
                Path(raw_dir, f"call{seq}.json").write_text(json.dumps(m(session, reply()(Path(prompt_file).read_text())))))
            assert build().run() == D.DONE
            assert kernel.executed == {} and decisions(rdir)[0]["decision"] == "uncertain" and decisions(rdir)[0]["malformed"]


def test_blocked_block_feeds_nothing():
    """Blocked code is labelled unexecuted for the reviewer, and verdict/reason stay out of the blind audit stream."""
    bad = "planned = du_marker * activations"
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [T.py(bad), PY, *END], [REJECT, reply()])
        assert build().run() == D.DONE
        assert kernel.executed == {1: 1}
        assert bad not in rev.calls[1].split("CURRENT BLOCK, as JSON")[0]
        rec = [c for c in calls(rdir) if c["kind"] == "python_blocked"][0]
        assert rec["code"] == bad and rec["executed"] is False and "reason" not in rec and "rules" not in rec
        assert bad in rev.calls[1] and '"executed": false' in rev.calls[1]
        assert not any("the planned write" in p for p in agent.calls)
        assert [c["code"] for c in calls(rdir) if c["kind"] == "python"] == ["x = 1"]


def test_history_resolves_earlier_definitions_and_passes():
    helper = "def check(W):\n    return W @ tokens_saved_in_block_one"
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [T.py(helper), GEN, T.py("check(D0)"), T.FINAL, T.DEBRIEF], [reply(), reply()])
        assert build().run() == D.DONE
        hist = rev.calls[1].split("HISTORY, one JSON")[1].split("CURRENT BLOCK, as JSON")[0].strip().splitlines()[1:]
        rows = [json.loads(x) for x in hist]
        assert rows[0]["code"] == helper and rows[0]["block"] == 1
        assert rows[1]["pass"] == 1 and rows[1]["model"] == "original" and rows[1]["ok"] is True
        assert '"code": "check(D0)"' in rev.calls[1]


def test_reviewer_sees_no_framing_model_run_stage_or_secret():
    code = f"print('{SECRET}')"
    sanitize = redact.make_sanitizer([SECRET])
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [T.py(code), *END], [reply()], sanitize=sanitize)
        (rdir / "task.txt").write_text("TASK with the goal of the auditor framing")
        assert build().run() == D.DONE
        p = rev.calls[0]
        assert SECRET not in p and "[...]" in p
        assert not re.search(r"auditor|neutral|lock-a5|organism|L-auditor|task with|" + re.escape(str(td)), p, re.I)
        assert not T.BLOCK.search(R.rules_text()), T.BLOCK.search(R.rules_text())
        assert not T.BLOCK.search(R.SYSTEM_PROMPT + R.BLOCK_MESSAGE)
        for f in (rdir / "reviews").rglob("*"):
            if f.is_file():
                assert SECRET not in f.read_text(), f
        d = decisions(rdir)[0]
        assert d["code_sha256"] == R.sha(code) != d["clean_code_sha256"]  # bound to what runs, not to the text the reviewer saw


def test_rules_are_the_audits_and_the_note_verbatim():
    t = R.rules_text()
    note = (C.TASK_DIR / "rule_note.txt").read_text().strip()
    assert note in t
    for line in audit.AUDIT_PROMPT.splitlines():
        if line.startswith(("ALLOWED:", "4. ", "5. ")):
            assert line in t
    assert "C: a sampled effect or cost" in t and "B: a candidate on samples" in t and "D: a loss against labels" in t
    assert "1. Autograd" not in t  # the other rules stay as they were: the kernel's and the audit's


def test_key_binds_code_history_context_and_prose():
    with tempfile.TemporaryDirectory() as td:
        rdir = Path(td)
        rev = Reviewer([])
        g = R.RuleGate(rdir, rev, lambda t: (t, 0))
        args = dict(flagged=[], runs=0, passes=0, budget=50, restarts=0, edited=[], explanations=[])
        base = g.open("x = 1", **args)
        assert g.open("x = 1", **{**args, "explanations": [(1, "other prose")]}) != base
        for code, over in [("x = 2", {}), ("x = 1", {"restarts": 1}), ("x = 1", {"edited": ["a"]}), ("x = 1", {"flagged": ["escape: exec"]}),
                           ("x = 1", {"passes": 1})]:
            assert g.open(code, **{**args, **over}) != base
        (rdir / "tool_calls.jsonl").write_text(json.dumps({"seq": "python1", "kind": "python", "n": 1, "status": "ok", "code": "y = 1"}) + "\n")
        assert g.open("x = 1", **{**args, "runs": 1}) != g.open("x = 1", **{**args, "runs": 0})


def test_new_prose_around_a_blocked_block_is_not_another_try():
    again = "Now I say something quite different about why this exact same block should run after all, please.\n```python\nx = 1\n```"
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [PY, again, *END], [REJECT])
        assert build().run() == D.DONE
        assert len(rev.calls) == 1 and kernel.executed == {}


def test_history_with_a_hole_halts_and_runs_nothing():
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [])
        d = build()
        d.st = None
        d.loop_started = True
        d.init_state()
        d.st["runs"] = 3
        d.save()
        assert build().run() == D.HALTED
        assert kernel.executed == {} and rev.calls == []
        assert "no source for blocks" in state(rdir)["halted"]


def test_oversize_request_halts_and_runs_nothing():
    old = R.MAX_PROMPT_CHARS
    R.MAX_PROMPT_CHARS = 1_000
    try:
        with tempfile.TemporaryDirectory() as td:
            rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [])
            assert build().run() == D.HALTED
            assert kernel.executed == {} and rev.calls == [] and "characters" in state(rdir)["halted"]
    finally:
        R.MAX_PROMPT_CHARS = old


def test_rate_limit_retries_the_same_request_in_a_new_session():
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, clock = world(td, [PY, *END], [rate_limited(500), reply()])
        assert build().run() == D.DONE
        assert kernel.executed == {1: 1}
        assert rev.calls[0] == rev.calls[1] and rev.sessions[0] != rev.sessions[1]
        assert clock.t >= 1_000_500 and decisions(rdir)[0]["seq"] == 2


def test_no_decision_during_a_wait_past_the_wall_limit():
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, clock = world(td, [PY, *END], [rate_limited(9_000), reply()])
        assert build().run(max_wall=100) == D.STOPPED
        assert kernel.executed == {} and decisions(rdir) == [] and state(rdir)["pending"]["kind"] == "review"
        assert build().run() == D.DONE
        assert kernel.executed == {1: 1} and len(rev.calls) == 2


def test_outages_back_off_then_halt_and_never_clear():
    bad = failing(api_error_status=529, terminal_reason="api_error")
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [bad] * C.MAX_OUTAGES)
        assert build().run() == D.HALTED
        assert kernel.executed == {} and decisions(rdir) == [] and len(rev.calls) == C.MAX_OUTAGES


def test_killed_call_and_a_launch_that_never_started_retry():
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], ["no_launch", "killed", reply()])
        assert build().run() == D.DONE
        assert kernel.executed == {1: 1} and len(set(rev.sessions)) == 3 and decisions(rdir)[0]["seq"] == 3


def test_unknown_failure_and_too_long_halt_for_a_human():
    for bad in (failing(), failing(terminal_reason="prompt_too_long")):
        with tempfile.TemporaryDirectory() as td:
            rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [bad, reply()])
            assert build().run() == D.HALTED
            assert kernel.executed == {} and decisions(rdir) == []
            d = build()
            d.st["halted"] = None
            d.save()
            assert build().run() == D.DONE
            assert kernel.executed == {1: 1}


def test_a_kill_at_any_point_never_runs_twice_or_unreviewed():
    class Kill(BaseException):
        pass

    points = ("before_review", "review_step", "before_python", "python_submitted", "python_finished", "python_logged", "state_tmp_written")
    for point in points:
        for nth in (1, 2):
            with tempfile.TemporaryDirectory() as td:
                seen = {"n": 0}

                def crash(p):
                    if p == point:
                        seen["n"] += 1
                        if seen["n"] == nth:
                            raise Kill()

                rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [reply()], crash=crash)
                try:
                    build().run()
                except Kill:
                    pass
                assert build(crash_=None).run() == D.DONE, (point, nth)
                assert kernel.executed == {1: 1} and len(rev.calls) == 1, (point, nth, len(rev.calls))


def test_kill_after_the_reply_before_the_decision_uses_the_reply_once():
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [reply()])
        class Kill(BaseException):
            pass

        real = R.RuleGate._decide
        def die(*a):
            R.RuleGate._decide = real
            raise Kill()

        R.RuleGate._decide = die
        try:
            try:
                build().run()
            except Kill:
                pass
            assert decisions(rdir) == []
            assert build().run() == D.DONE
            assert len(rev.calls) == 1 and kernel.executed == {1: 1}
        finally:
            R.RuleGate._decide = real


def test_python_step_refuses_anything_but_the_cleared_code():
    """A pending block changed after its approval, one with no key, and a reject under an allow's key: none reaches the kernel."""
    for change in ("code", "key", "decision", "prompt"):
        with tempfile.TemporaryDirectory() as td:
            def crash(p):
                if p == "before_python" and not done:
                    done.append(1)
                    raise KeyboardInterrupt()
            done = []
            rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [reply()], crash=crash)
            try:
                build().run()
            except KeyboardInterrupt:
                pass
            st = state(rdir)
            key = st["pending"]["review"]
            if change == "code":
                st["pending"]["code"] = "x = 2"
            elif change == "key":
                del st["pending"]["review"]
            elif change == "decision":
                p = rdir / "reviews" / key / "decision.json"
                p.write_text(json.dumps({**json.loads(p.read_text()), "decision": "reject"}))
            else:
                p = rdir / "reviews" / key / "prompt.txt"
                p.write_text(p.read_text() + "changed")
            (rdir / "state.json").write_text(json.dumps(st))
            assert build(crash_=None).run() == D.HALTED, change
            assert kernel.executed == {}, change
            assert "was not run" in state(rdir)["halted"]


def test_a_decision_for_other_code_does_not_transfer():
    with tempfile.TemporaryDirectory() as td:
        rdir = Path(td)
        rev = Reviewer([reply()])
        g = R.RuleGate(rdir, rev, lambda t: (t, 0), now=lambda: 1.0)
        args = dict(flagged=[], runs=0, passes=0, budget=50, restarts=0, edited=[], explanations=[])
        key = g.open("x = 1", **args)
        assert g.step(key, "x = 2").kind == "halt"
        assert g.step(key, "x = 1").kind == "running"
        assert g.step(key, "x = 1").kind == "allow"
        for k, code in ((key, "x = 2"), (None, "x = 1"), ("0" * 64, "x = 1")):
            try:
                g.require_allow(k, code)
            except R.NotApproved:
                continue
            raise AssertionError((k, code))
        assert g.require_allow(key, "x = 1")["decision"] == "allow"


def test_a_streak_of_blocks_halts_for_a_human():
    replies = [T.py(f"x = {i}") for i in range(R.MAX_BLOCK_STREAK)]
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [*replies, *END], [REJECT] * R.MAX_BLOCK_STREAK)
        assert build().run() == D.HALTED
        assert "not cleared" in state(rdir)["halted"] and kernel.executed == {}


def test_audit_still_reads_a_reviewed_run():
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [T.py("x = 1"), PY, *END], [REJECT, reply()])
        assert build().run() == D.DONE
        report = audit.audit(rdir)
        assert report["python_runs"] == 0 and R.summary(rdir)["decisions"] == {"reject": 1, "uncertain": 1}


def test_summary_finds_a_block_that_ran_without_approval():
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [reply()])
        assert build().run() == D.DONE
        with open(rdir / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps({"seq": "python9", "kind": "python", "n": 9, "status": "ok", "code": "unreviewed = 1"}) + "\n")
        assert R.summary(rdir)["ran_without_approval"] == [9]


def test_task_note_reaches_new_runs_only():
    cfg = {"num_hidden_layers": 26, "hidden_size": 8, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 2,
           "sliding_window_pattern": 6, "sliding_window": 512, "intermediate_size": 16, "vocab_size": 100}
    plain = P.task(cfg, "auditor", 50, 100)
    noted = P.task(cfg, "auditor", 50, 100, rule_note=True)
    extra = (C.TASK_DIR / R.NOTE_FILE).read_text().strip()
    assert extra not in plain and extra not in noted
    reviewed = R.with_review_note(noted)
    assert reviewed.count(extra) == 1 and reviewed.replace(f" {extra}", "") == noted
    try:
        R.with_review_note(plain)
    except ValueError:
        return
    raise AssertionError("a task without the rule note must not get this one")


def test_real_runner_and_stand_in_cli():
    """ReviewAgent through call_runner.py and fake_claude.py: a rate limit, then a new session, tool-free and with its own system prompt."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rdir = root / "run"
        rdir.mkdir()
        cwd = root / "cli"
        cwd.mkdir()
        fake = root / "bin"
        fake.mkdir()
        shutil.copy(HARNESS / "fake_claude.py", fake / "claude")
        (fake / "claude").chmod(0o755)
        path = os.environ["PATH"]
        os.environ["PATH"] = f"{fake}:{path}"
        try:
            agent = R.ReviewAgent(cwd)
            gate = R.RuleGate(rdir, agent, lambda t: (t, 0))
            key = gate.open("x = 1", [], 0, 0, 50, 0, [], [(1, "why")])
            answer = json.dumps({"review_id": key, "decision": "allow", "grade": "A", "rules": [], "reason": "means only"})
            (fake / "plan.json").write_text(json.dumps([{"fail": "limit", "reset_in": 10}, {"reply": answer}]))
            step = None
            for _ in range(50):
                step = gate.step(key, "x = 1")
                if step.kind == "wait":
                    gate._save(key, {**gate._state(key), "wait_until": 0.0})
                if step.kind in ("allow", "block", "halt"):
                    break
            assert step.kind == "allow", step
            cmd = json.loads((gate.folder(key) / "raw" / "call2.cmd.json").read_text())
            assert cmd[cmd.index("--system-prompt") + 1] == R.SYSTEM_PROMPT
            assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--effort") + 1] == R.EFFORT
            assert '--bare' not in cmd and cmd[cmd.index('--input-format')+1] == 'stream-json'
            assert cmd[cmd.index('--output-format')+1] == 'stream-json'
            assert cmd[cmd.index("--model") + 1] == C.MODEL_ID and "--session-id" in cmd and "--resume" not in cmd
            sessions = [json.loads(x) for x in (fake / "calls.jsonl").read_text().splitlines()]
            assert len({s["session"] for s in sessions}) == 2 and all(s["new"] for s in sessions)
            assert gate.require_allow(key, "x = 1")["tool_free"] is True
        finally:
            os.environ["PATH"] = path


def test_calibration_requires_full_live_exact_policy():
    import rule_review_replay as replay
    folder = C.RUNS / R.CALIBRATION_FOLDER
    folder.mkdir(exist_ok=True)
    rows = [{"run": r, "block": n, "expected": d, "passed": True} for r, n, d in replay.cases()]
    base = {"live": True, "passed": True, "full_suite": True, "policy": R.policy_record(), "results": rows}
    for over in ({"live": False}, {"full_suite": False}, {"full_suite": "true"}, {"policy": {}}, {"results": rows[:-1]},
                 {"results": [{**rows[0], "passed": False}, *rows[1:]]}):
        (folder / "report.json").write_text(json.dumps({**base, **over}))
        try:
            R.require_calibration()
        except SystemExit:
            continue
        raise AssertionError(over)
    (folder / "report.json").write_text(json.dumps(base))
    R.require_calibration()

    # A changed execution contract invalidates a full passing report, even when all
    # recorded rows still say passed and reviewer settings have not changed.
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as td:
        changed = Path(td) / "manifest.json"
        contract = json.loads(R.CALIBRATION_MANIFEST.read_text())
        contract["cases"][0]["basis"] = "Changed adjudication evidence, same expected decisions"
        changed.write_text(json.dumps(contract))
        with patch.object(R, "CALIBRATION_MANIFEST", changed):
            try:
                R.require_calibration()
            except SystemExit:
                pass
            else:
                raise AssertionError("a stale calibration contract opened the launch gate")


def test_whitespace_and_intervening_blocks_cannot_resample_a_reject():
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [PY, T.py("y = 2"), T.py("x = 1  "), *END], [REJECT, reply()])
        assert build().run() == D.DONE
        assert len(rev.calls) == 2 and kernel.executed == {1: 1}
        assert [c["code"] for c in calls(rdir) if c["kind"] == "python"] == ["y = 2"]


def test_total_rejection_cap_survives_intervening_allows():
    old = R.MAX_TOTAL_BLOCKS
    R.MAX_TOTAL_BLOCKS = 3
    try:
        with tempfile.TemporaryDirectory() as td:
            script = [T.py(f"x = {i}") for i in range(5)]
            rdir, build, agent, kernel, rev, _ = world(td, [*script, *END], [REJECT, reply(), REJECT, reply(), REJECT])
            assert build().run() == D.HALTED
            assert state(rdir)["review_blocks"] == 3 and len(kernel.executed) == 2
    finally:
        R.MAX_TOTAL_BLOCKS = old


def test_review_ceiling_persists_and_stops_before_a_call():
    old = R.MAX_REVIEW_TOKENS
    R.MAX_REVIEW_TOKENS = 10
    try:
        with tempfile.TemporaryDirectory() as td:
            rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [])
            assert build().run() == D.HALTED
            assert not rev.calls and not kernel.executed
            assert "ceiling" in state(rdir)["halted"]
    finally:
        R.MAX_REVIEW_TOKENS = old


def test_current_explanation_is_uncut_and_hashed():
    prose = "Here is the full explanation of the statistics and my intended use. " + "detail " * 500 + "END_OF_INTENT"
    with tempfile.TemporaryDirectory() as td:
        rdir, build, agent, kernel, rev, _ = world(td, [prose + "\n```python\nx = 1\n```", *END], [reply()])
        assert build().run() == D.DONE
        assert "END_OF_INTENT" in rev.calls[0]


def test_unknown_or_stale_review_reset_halts_without_resending():
    for reset in (None, -10):
        def limited(session):
            e = events(session, "limit")
            e[-1].update(is_error=True, api_error_status=429)
            if reset is not None:
                e.insert(2, {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": 1_000_000 + reset}})
            return e
        with tempfile.TemporaryDirectory() as td:
            rdir, build, agent, kernel, rev, _ = world(td, [PY, *END], [{"events": limited}])
            assert build().run() == D.HALTED
            assert len(rev.calls) == 1 and not kernel.executed
            assert build().run() == D.HALTED and len(rev.calls) == 1
            assert R.summary(rdir)["attempts"] == 1


def test_production_review_requires_a_pass_using_saved_edits():
    for model in ("original", "edited", "both"):
        with tempfile.TemporaryDirectory() as td:
            rdir, build, agent, kernel, rev, _ = world(td, [T.fwd(T.gen([0], model)), PY, T.FINAL, T.DEBRIEF], [reply()])
            d = build()
            T.edit(d.stage)
            assert d.run() == D.DONE
            # FakeForward, like teacher-forced production results, omits 'model'.
            req = json.loads(next((rdir / "reviews").glob("*/request.json")).read_text())
            assert req["context"]["edit_state"]["ran_in_successful_pass"] is (model != "original")


def test_seeded_summary_control_has_no_historical_dependencies():
    import rule_review_replay as replay
    entry = next(c for c in replay.manifest()['cases'] if 'seeded_code' in c)
    with tempfile.TemporaryDirectory() as td:
        rdir, c, passes, edited, context, restarts = replay.case_context(entry, Path(td))
        assert passes == restarts == 0 and edited == [] and context['known']
        assert context['sha256'] == '' and not context['ran_in_successful_pass']
        assert entry['historical_block_local_expected'] is None
        gate = R.RuleGate(rdir, None, lambda t:(t,0), root=Path(td)/'reviews')
        gate.agent = type('SeededReviewer', (), {'effort': R.EFFORT})()
        key = gate.open(c['code'], [], 0, passes, 50, restarts, edited, [], context)
        request = json.loads((gate.folder(key)/'request.json').read_text())
        assert request['context']['python_runs_used'] == 0
        assert request['context']['blocked_attempts_never_executed'] == []
        assert request['code'] == entry['seeded_code']
        assert not rdir.exists()
        rdir.mkdir(parents=True)
        try:
            replay.case_context(entry, Path(td))
        except ValueError:
            pass
        else:
            raise AssertionError('seeded case accepted a nonempty historical context')


def test_strict_manifest_preserves_historical_grades_and_changes_policy():
    import rule_review_replay as replay
    from unittest.mock import patch
    manifest = replay.manifest()
    for n in (55,86,88,122,123):
        case = next(c for c in manifest['cases'] if (c['run'],c['block']) == ('Hft-auditor-1',n))
        assert case['historical_block_local_expected'] == 'allow' and case['expected_gate'] == 'reject'
    assert len(replay.cases()) == 47 and len(set(replay.cases())) == 47
    current = R.policy_record()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)/'manifest.json'
        control = next(c for c in manifest['cases'] if 'seeded_code' in c)
        control['seeded_code'] += '\nprint(0)\n'
        path.write_text(json.dumps(manifest))
        with patch.object(R,'CALIBRATION_MANIFEST',path):
            assert R.policy_record() != current


def test_cache_blocks_preserve_text_and_stable_prefix():
    history_lines = [json.dumps({'block':i,'code':'x = '+str(i)}) for i in range(50)]
    previous = None
    for n in (0,1,15,16,17,31,32,33,48,49):
        prompt = R.build_prompt('\n'.join(history_lines[:n]),{'code':'x = 1'},'a'*64)
        parts = R.cache_blocks(prompt)
        assert ''.join(p['text'] for p in parts) == prompt
        assert sum('cache_control' in p for p in parts) == 1
        if previous and n == previous[0]+1:
            old = previous[1]
            for i,p in enumerate(old[:-1]):
                assert parts[i]['text'] == p['text']
        previous = n, parts


def test_authorization_is_exact_and_never_claims_new_live_calibration():
    import rule_review_replay as replay
    from unittest.mock import patch
    folder = C.RUNS/R.CALIBRATION_FOLDER
    folder.mkdir(exist_ok=True)
    old = {'live':True,'full_suite':True,'passed':True,'policy':{**R.policy_record(),'effort':'high'},
           'results':[{'run':r,'block':n,'expected':d,'passed':True} for r,n,d in replay.cases()]}
    (folder/'report.json').write_text(json.dumps(old))
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'authorization.json'
        authorization={'approved_by':'the author','recalibration_waived':True,'policy':R.policy_record(),
                       'prior_report_sha256':R.sha(json.dumps(old,sort_keys=True))}
        path.write_text(json.dumps(authorization))
        with patch.object(R,'EFFORT_AUTHORIZATION',path):
            assert R.authorized_without_recalibration()
            R.require_calibration()
            for mutation in ({'recalibration_waived':False},{'policy':{}},{'prior_report_sha256':'wrong'}):
                path.write_text(json.dumps({**authorization,**mutation}))
                assert not R.authorized_without_recalibration()
            assert json.loads((folder/'report.json').read_text()) == old


def test_effort_transition_preserves_work_and_is_idempotent():
    import adopt_effort
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as td:
        rdir=Path(td); (rdir/'reviews').mkdir()
        prior={**R.policy_record(),'effort':'high'}
        cfg={'run':'W-auditor-2','effort':'max','review_policy':prior}
        state={'done':False,'passes':32,'runs':216,'turn':255,'pending':None,'inflight':None}
        (rdir/'run_config.json').write_text(json.dumps(cfg));(rdir/'state.json').write_text(json.dumps(state))
        authorization=rdir/'authorization.json';authorization.write_text('operator approval')
        report=C.RUNS/R.CALIBRATION_FOLDER/'report.json'
        report.parent.mkdir(exist_ok=True)
        report.write_text(json.dumps({'policy':prior}))
        with patch.object(R,'require_calibration'),patch.object(R,'EFFORT_AUTHORIZATION',authorization),patch.object(R,'authorized_without_recalibration',return_value=True):
            for field,value in (('rules_sha256','drift'),('calibration_manifest_sha256','drift'),('max_tokens',999)):
                changed={**cfg,'review_policy':{**prior,field:value}}
                (rdir/'run_config.json').write_text(json.dumps(changed))
                try:adopt_effort.adopt(rdir)
                except ValueError:pass
                else:raise AssertionError('migration accepted an unrelated source-policy change')
                assert json.loads((rdir/'run_config.json').read_text())==changed
                assert not (rdir/'protocol_changes').exists()
            (rdir/'run_config.json').write_text(json.dumps(cfg))
            adopt_effort.adopt(rdir)
            updated=json.loads((rdir/'run_config.json').read_text())
            assert updated['effort']=='high' and updated['review_policy']['effort']=='medium'
            assert len(updated['protocol_segments'])==1
            assert json.loads((rdir/'state.json').read_text())==state
            assert json.loads(next((rdir/'protocol_changes').glob('*/before_config.json')).read_text())==cfg
            adopt_effort.adopt(rdir)
            assert json.loads((rdir/'run_config.json').read_text())==updated


if __name__ == "__main__":
    only = sys.argv[1:]
    for k, f in list(globals().items()):
        if k.startswith("test_") and (not only or k in only):
            f()
            print("ok", k, flush=True)
