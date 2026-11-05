#!/usr/bin/env python3
"""A stand-in for the `claude` binary: no model, no network. It keeps Claude Code 2.1.317's session rules and event shapes:
--session-id on a used id and --resume on an unknown id fail before the prompt is read, with the CLI's exact messages; any
other call records its prompt in the session first, as the CLI does, whether or not a reply follows.

Installed as a file named `claude` in a folder put first on PATH. Next to it: plan.json, a list with one entry per call that
gets past the session check, each {"reply": text}, {"fail": kind} or {"sleep": s, "reply": text}. sessions/<id>.jsonl records
the prompts each conversation received; calls.jsonl records every invocation.
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SESSIONS = HERE / "sessions"


def events(session: str, text: str, **result) -> list[dict]:
    r = {"type": "result", "subtype": "success", "is_error": False, "result": text, "session_id": session, "stop_reason": "end_turn",
         "terminal_reason": "completed", "api_error_status": None, "usage": {"input_tokens": 1, "output_tokens": 1}, "total_cost_usd": 0.0,
         "num_turns": 1}
    r.update(result)
    return [{"type": "system", "subtype": "init", "session_id": session, "tools": [], "mcp_servers": [], "model": "claude-opus-5-5",
             "apiKeySource": "none", "claude_code_version": "2.1.317"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}, r]


def main() -> None:
    args = sys.argv[1:]
    if "--version" in args:
        print("2.1.317 (fake)")
        return
    new = args[args.index("--session-id") + 1] if "--session-id" in args else None
    old = args[args.index("--resume") + 1] if "--resume" in args else None
    sid = new or old
    SESSIONS.mkdir(exist_ok=True)
    log = SESSIONS / f"{sid}.jsonl"
    with open(HERE / "calls.jsonl", "a") as f:
        f.write(json.dumps({"session": sid, "new": bool(new), "t": time.time()}) + "\n")
    if new and log.exists():
        print(f"Error: Session ID {new} is already in use.", file=sys.stderr)
        sys.exit(1)
    if old and not log.exists():
        print(f"No conversation found with session ID: {old}", file=sys.stderr)
        sys.exit(1)
    prompt = sys.stdin.read()
    with open(log, "a") as f:
        f.write(json.dumps(prompt) + "\n")
    with open(HERE / "plan.lock", "a+") as lk:
        import fcntl

        fcntl.flock(lk, fcntl.LOCK_EX)
        counter = HERE / "plan.index"
        i = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(i + 1))
    plan = json.loads((HERE / "plan.json").read_text())
    step = plan[i] if i < len(plan) else {"fail": "exhausted"}
    time.sleep(step.get("sleep", 0))
    fail = step.get("fail")
    if fail is None:
        print(json.dumps(events(sid, step["reply"])))
        if step.get("forget"):  # the conversation is gone: the next --resume fails, as after a cleared store
            log.unlink()
    elif fail == "limit":  # 429, as the CLI wraps it: a rejected window with its reset time, and a text result
        ev = events(sid, "You've hit your session limit · resets 5pm", is_error=True, api_error_status=429, terminal_reason="api_error")
        ev.insert(2, {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": int(time.time()) + step.get("reset_in", 1),
                                                                     "rateLimitType": "five_hour"}})
        ev[1].update(error="rate_limit", api_error="usage_limit_reached", api_error_status=429)
        print(json.dumps(ev))
        sys.exit(1)
    elif fail == "overloaded":
        ev = events(sid, "API Error: Repeated 529 Overloaded errors", is_error=True, api_error_status=529, terminal_reason="api_error")
        print(json.dumps(ev))
        sys.exit(1)
    elif fail == "refusal":
        print(json.dumps(events(sid, "", stop_reason="refusal", is_error=True, terminal_reason="model_error")))
        sys.exit(1)
    elif fail == "unknown":
        print("Error: something this harness has never seen", file=sys.stderr)
        sys.exit(1)
    elif fail == "die":  # the process vanishes with the prompt recorded and no reply
        os.kill(os.getpid(), 9)
    else:
        print(f"unexpected fake failure {fail}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
