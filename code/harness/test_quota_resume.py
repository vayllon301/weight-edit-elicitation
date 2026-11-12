"""Quota boundary checks with seeded events, clock and tiny model; no provider or real-model calls."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testenv
import test_harness as T
import agent as A
import driver as D
import rule_review_replay as replay


def test_rejected_event_overrides_nominal_success_and_partial_python():
    e = A.scripted_events("s", T.py("danger = 1"))
    e.insert(2, {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": 2_000_000}})
    r = A.classify(e, "", 0, 1_000_000)
    assert r.kind == "rate_limit" and not r.text and r.wait == 1_000_060
    assert A.classify(e, "", -9, 1_000_000, timed_out=True).kind == "rate_limit"


def test_all_rejected_windows_are_respected_even_after_an_allowed_event():
    e = A.scripted_events("s", "partial")
    for t, reset in (("five_hour", 2_000), ("seven_day", 9_000)):
        e.insert(1, {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": reset, "rateLimitType": t}})
    e.insert(-1, {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "resetsAt": 1_500}})
    r = A.classify(e, "", 0, 1_000)
    assert r.kind == "rate_limit" and r.wait == 8_060


def test_missing_stale_and_nonfinite_resets_never_trigger_polling():
    for reset in (None, 999, float("nan"), float("inf"), "2000", True):
        e = A.scripted_events("s", "limit")
        e[-1].update(is_error=True, api_error_status=429)
        e.insert(1, {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": reset}})
        r = A.classify(e, "", 1, 1_000)
        assert r.kind == "rate_limit" and r.wait == 0, reset


def test_unknown_reset_is_durable_and_no_agent_block_executes():
    with tempfile.TemporaryDirectory() as td:
        rdir, cfg, agent, kernel, fwd = T.setup(Path(td), 1, [T.py("x = 1")])
        clock = T.Clock()
        d = T.make(rdir, cfg, agent, kernel, fwd, clock)
        d.init_state()
        d.failed_call(A.Result("rate_limit", wait=0))
        assert T.make(rdir, cfg, agent, kernel, fwd, clock).run() == D.HALTED
        assert not agent.calls and not kernel.executed
        assert "reset missing/stale" in json.loads((rdir / "state.json").read_text())["halted"]


def test_repeated_future_resets_are_bounded_without_a_successful_reply():
    with tempfile.TemporaryDirectory() as td:
        rdir, cfg, agent, kernel, fwd = T.setup(Path(td), 1, [T.py("x = 1")])
        d = T.make(rdir, cfg, agent, kernel, fwd, T.Clock())
        d.init_state()
        for _ in range(4):
            d.failed_call(A.Result("rate_limit", wait=100))
        assert d.st["halted"] and d.st["quota_stops"] == 4


def test_clock_reset_respects_explicit_zone_and_abstains_on_dates():
    assert A.limit_wait("You've hit your limit · resets 5pm (America/Chicago)", 1_800_000_000) > 60
    assert A.limit_wait("resets Nov 12 at 5pm (America/Chicago)", 1_800_000_000) == 0
    assert A.limit_wait("resets 99pm", 1_800_000_000) == 0
    assert A.limit_wait("resets 5pm (Not/AZone)", 1_800_000_000) == 0


def test_historical_replay_does_not_invent_zero_passes_or_no_edits():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        calls = [{"kind": "python", "n": 1, "code": "a = 1"},
                 {"kind": "forward", "k": 1, "edit_sha256": "edited-sha", "result": {"ok": True, "model": "both"}},
                 {"kind": "python", "n": 2, "code": "load(name)"},
                 {"kind": "python", "n": 3, "code": "inspect(a)"}]
        (p / "tool_calls.jsonl").write_text("\n".join(map(json.dumps, calls)))
        (p / "candidates.jsonl").write_text(json.dumps({"at": "pass-request-1", "dir": "edited", "tensor": "tensor-a"}))
        (p / "requests").mkdir()
        (p / "requests" / "fwd_1.json").write_text(json.dumps({"model": "both"}))
        c, count, edits, ctx, _ = replay.context_before(p, 2)
        assert count == 1 and edits == ["tensor-a"] and ctx["sha256"] == "edited-sha" and ctx["ran_in_successful_pass"]
        assert replay.context_before(p, 3)[3] == {"known": False, "sha256": None, "ran_in_successful_pass": None}


def test_replay_uses_request_model_and_abstains_without_it():
    for model, ok, expected in (("original", True, False), ("edited", True, True),
                                ("both", True, True), ("both", False, False), (None, True, False)):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            # Teacher-forced results have no model, even when the saved edit hash is present.
            calls = [{"kind": "forward", "k": 1, "edit_sha256": "edited-sha", "result": {"ok": ok}},
                     {"kind": "python", "n": 1, "code": "inspect(a)"}]
            (p / "tool_calls.jsonl").write_text("\n".join(map(json.dumps, calls)))
            (p / "candidates.jsonl").write_text(json.dumps({"at": "pass-request-1", "dir": "edited", "tensor": "tensor-a"}))
            (p / "requests").mkdir()
            if model:
                (p / "requests" / "fwd_1.json").write_text(json.dumps({"model": model}))
            assert replay.context_before(p, 1)[3]["ran_in_successful_pass"] is expected


def test_discarded_quota_usage_survives_crashes_exactly_once():
    for point in ("failed_reply_saved", "state_tmp_written", None):
        with tempfile.TemporaryDirectory() as td:
            rdir, cfg, agent, kernel, fwd = T.setup(Path(td), 1, [T.py("must_not_run = 1")])
            clock = T.Clock()
            d = T.make(rdir, cfg, agent, kernel, fwd, clock)
            d.init_state()
            d.ask()
            h = d.st["inflight"]
            raw = d.tdir(h["turn"]) / "raw" / f"call{h['seq']}.json"
            events = json.loads(raw.read_text())
            events[-1].update(usage={"input_tokens": 10_000, "output_tokens": 12}, total_cost_usd=0.5)
            events.insert(1, {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}})
            raw.write_text(json.dumps(events))

            def crash(p):
                if point and p == point:
                    raise T.Crash()

            d.crash = crash
            try:
                d.finish_call()
            except T.Crash:
                pass
            assert T.make(rdir, cfg, agent, kernel, fwd, clock).run() == D.HALTED
            assert T.make(rdir, cfg, agent, kernel, fwd, clock).run() == D.HALTED
            usage = json.loads((rdir / "state.json").read_text())["usage"]
            assert usage["input"] == 10_000 and usage["output"] == 12 and usage["cost_usd"] == 0.5
            assert not kernel.executed


def test_missing_failed_cost_does_not_erase_cumulative_conversation_cost():
    with tempfile.TemporaryDirectory() as td:
        rdir, cfg, agent, kernel, fwd = T.setup(Path(td), 1, [])
        d = T.make(rdir, cfg, agent, kernel, fwd)
        d.init_state()
        d.add_usage({"total_cost_usd": 0.5})
        for audit in ({}, {"total_cost_usd": 0.1}, {"total_cost_usd": 0.7}):
            d.failed_call(A.Result("rate_limit", audit=audit))
        assert d.st["usage"]["cost_usd"] == 0.7


def test_first_call_quota_retry_counts_independent_sessions_after_resume():
    with tempfile.TemporaryDirectory() as td:
        rdir, cfg, agent, kernel, fwd = T.setup(Path(td), 1, [T.py("x = 1")])
        clock = T.Clock()
        d = T.make(rdir, cfg, agent, kernel, fwd, clock)
        d.init_state()
        d.ask()
        first_session = d.st["session"]
        d.failed_call(A.Result("rate_limit", wait=100, audit={"usage": {"output_tokens": 500}, "total_cost_usd": 5.0}))
        clock.sleep(100)
        d = T.make(rdir, cfg, agent, kernel, fwd, clock)
        d.st = json.loads((rdir / "state.json").read_text())
        d.ask()
        assert d.st["session"] != first_session and d.st["conversation"] == 1
        h = d.st["inflight"]
        raw = d.tdir(h["turn"]) / "raw" / f"call{h['seq']}.json"
        events = json.loads(raw.read_text())
        events[-1].update(usage={"output_tokens": 300}, total_cost_usd=3.0)
        raw.write_text(json.dumps(events))
        d.finish_call()
        usage = json.loads((rdir / "state.json").read_text())["usage"]
        assert usage["cost_usd"] == 8.0 and usage["output"] == 800


def test_session_cost_migrates_existing_conversation_totals_without_double_counting():
    with tempfile.TemporaryDirectory() as td:
        rdir, cfg, agent, kernel, fwd = T.setup(Path(td), 1, [])
        d = T.make(rdir, cfg, agent, kernel, fwd)
        d.init_state()
        d.st["conversation"] = 2
        d.st["usage"].update(cost_usd=7.0, cost_by_conversation={"1": 2.0, "2": 5.0})
        d.add_usage({"total_cost_usd": 6.0})
        d.add_usage({})
        assert d.st["usage"]["cost_usd"] == 8.0
        d.st["session"] = "a-new-cli-session"
        d.add_usage({"total_cost_usd": 3.0})
        d.add_usage({"total_cost_usd": 4.0})
        assert d.st["usage"]["cost_usd"] == 12.0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k, flush=True)
