"""One `claude -p` call per turn, through call_runner.py, which outlives the driver: its reply lands in a file, so a
killed driver can pick the reply up when it comes back. ScriptedAgent replays canned replies through the same files."""

import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import config as C

# Failure kinds, from Claude Code 2.1.317's own code and result schema (structured fields, not message wording):
#   result.terminal_reason, result.stop_reason, result.api_error_status, assistant.error, assistant.api_error,
#   rate_limit_event.rate_limit_info.status. The two session errors have no result event and no field: they are the
#   CLI's exact strings. Anything else is "unknown": the run stops cleanly and no prompt is sent again.
NO_SESSION = re.compile(r"^(Error: )?No conversation found with session ID: \S+\s*$", re.M)
SESSION_IN_USE = re.compile(r"^Error: Session ID \S+ is already in use\.\s*$", re.M)
TOO_LONG_REASONS = {"prompt_too_long", "blocking_limit"}
OUTAGE_API_ERRORS = {"connection_failed", "no_response", "stream_idle_timeout", "computer_slept", "empty_completion"}
OK_REASONS = {None, "completed"}
RESET_EPOCH = re.compile(r"\|(\d{10})\b")
RESET_CLOCK = re.compile(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.I)
DEFAULT_LIMIT_WAIT = 0.0  # unknown reset: halt for an explicit resume, never poll the subscription


@dataclass
class Result:
    kind: str  # ok, rate_limit, outage, refused, bad_session, too_long, interrupted, unknown
    text: str = ""
    audit: dict = field(default_factory=dict)
    wait: float = 0.0
    detail: str = ""


def parse_events(raw: str) -> list[dict]:
    """--output-format json with --verbose gives an array of events; stream-json gives one per line."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [e for e in (data if isinstance(data, list) else [data]) if isinstance(e, dict)]
    except ValueError:
        out = []
        for line in raw.splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict):
                out.append(e)
        return out


def audit_events(events: list[dict]) -> dict:
    """What the call did. tool_free holds only if the stream shows the call and no tool item in it."""
    types: dict[str, int] = {}
    blocks: dict[str, int] = {}
    tools: dict[str, int] = {}
    init, result, limit, api = {}, {}, {}, {}
    limits = []
    for e in events:
        t = str(e.get("type"))
        sub = e.get("subtype")
        key = f"{t}.{sub}" if sub else t
        types[key] = types.get(key, 0) + 1
        if t == "system" and sub == "init":
            init = e
        if t == "result":
            result = e
        if t == "rate_limit_event":
            limit = e.get("rate_limit_info") or {}
            limits.append(limit)
        if t == "assistant":
            for k in ("error", "api_error", "api_error_status"):
                if e.get(k) is not None:
                    api[k] = e[k]
        msg = e.get("message")
        for b in (msg.get("content") if isinstance(msg, dict) and isinstance(msg.get("content"), list) else []):
            if isinstance(b, dict):
                bt = str(b.get("type"))
                blocks[bt] = blocks.get(bt, 0) + 1
                if "tool" in bt:
                    tools[str(b.get("name", bt))] = tools.get(str(b.get("name", bt)), 0) + 1
    stream_seen = bool(init) and bool(result)
    init_tools = init.get("tools") or []
    init_mcp = init.get("mcp_servers") or []
    compacted = sum(n for k, n in types.items() if "compact" in k)
    return {"event_types": types, "item_types": blocks, "tool_items": tools,
            "tool_free": stream_seen and not tools and not init_tools and not init_mcp,
            "stream_seen": stream_seen, "init_tools": init_tools, "init_mcp_servers": init_mcp, "compactions": compacted,
            "model": init.get("model"), "session_id": result.get("session_id") or init.get("session_id"),
            "claude_code_version": init.get("claude_code_version"),
            "stop_reason": result.get("stop_reason"), "is_error": result.get("is_error"),
            "terminal_reason": result.get("terminal_reason"),
            "api_error_status": result.get("api_error_status") or api.get("api_error_status"),
            "assistant_error": api.get("error"), "api_error": api.get("api_error"), "result_seen": bool(result),
            "rate_limit": {k: limit.get(k) for k in ("status", "rateLimitType", "resetsAt", "unifiedWindows")} if limit else None,
            "rate_limits": limits,
            "usage": result.get("usage"), "model_usage": result.get("modelUsage"),
            "total_cost_usd": result.get("total_cost_usd"), "duration_ms": result.get("duration_ms"),
            "duration_api_ms": result.get("duration_api_ms"), "num_turns": result.get("num_turns")}


def reply_text(events: list[dict]) -> str:
    # A turn can end in a thinking block after its text; the CLI then reports an empty result and the text is in the assistant event.
    for e in reversed(events):
        if e.get("type") == "result" and isinstance(e.get("result"), str) and e["result"].strip():
            return e["result"]
    for e in reversed(events):
        msg = e.get("message")
        if e.get("type") == "assistant" and isinstance(msg, dict):
            parts = [b.get("text", "") for b in msg.get("content", []) if isinstance(b, dict) and b.get("type") == "text"]
            if parts:
                return "\n".join(parts)
    return ""


def limit_wait(text: str, now: float) -> float:
    m = RESET_EPOCH.search(text)
    if m:
        return int(m.group(1)) - now + 60 if int(m.group(1)) > now else 0.0
    m = RESET_CLOCK.search(text)
    if m:
        if not 1 <= int(m.group(1)) <= 12 or int(m.group(2) or 0) > 59:
            return 0.0
        # Date-bearing messages need structured metadata. Never interpret a weekly reset as tonight.
        if re.search(r"\b(mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b", text, re.I):
            return 0.0
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        zone = re.search(r"\(([A-Za-z_]+/[A-Za-z_/]+|UTC)\)", text)
        try:
            tz = ZoneInfo(zone.group(1)) if zone else datetime.fromtimestamp(now).astimezone().tzinfo
        except ZoneInfoNotFoundError:
            return 0.0
        dt = datetime.fromtimestamp(now, tz)
        target = dt.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
        if target.timestamp() <= now:
            target += timedelta(days=1)
        return target.timestamp() - now + 60
    return DEFAULT_LIMIT_WAIT


def quota_wait(audit: dict, text: str, now: float) -> float:
    """Only future reset epochs of rejected windows are authoritative. Ambiguous/stale resets stop the run."""
    limits = audit.get("rate_limits") or [audit.get("rate_limit") or {}]
    rejected = [x for x in limits if x.get("status") == "rejected"]
    resets = [x.get("resetsAt") for x in rejected]
    if resets:
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= now for x in resets):
            return 0.0
        return max(resets) - now + 60
    return limit_wait(text, now)


def classify(events: list[dict], stderr: str, rc: int | None, now: float, *, finished: bool = True,
             timed_out: bool = False) -> Result:
    """finished: the CLI's exit was recorded. Without it and without a result event, the call was killed."""
    audit = audit_events(events)
    text = reply_text(events)
    detail = f"rc={rc} status={audit['api_error_status']} terminal={audit['terminal_reason']} stop={audit['stop_reason']} " \
             f"error={audit['assistant_error']}/{audit['api_error']}: {(text[:120] + ' ' + stderr[-200:]).strip()}"
    status = audit["api_error_status"]
    limit = audit["rate_limit"] or {}
    quota = any(x.get("status") == "rejected" for x in audit.get("rate_limits", [])) or status == 429 \
        or audit["assistant_error"] == "rate_limit" or audit["api_error"] == "usage_limit_reached"
    if quota:
        return Result("rate_limit", audit=audit, wait=quota_wait(audit, text + stderr, now), detail=detail)
    if timed_out:
        return Result("outage", audit=audit, detail="the call ran past the time limit")
    if audit["stop_reason"] == "refusal":
        return Result("refused", audit=audit, detail=detail)
    failed = bool(audit["is_error"]) or not audit["result_seen"] or rc not in (0, None) \
        or audit["terminal_reason"] not in OK_REASONS
    if failed:
        if isinstance(status, int) and 500 <= status <= 599 or audit["api_error"] in OUTAGE_API_ERRORS:
            return Result("outage", audit=audit, detail=detail)
        if audit["terminal_reason"] in TOO_LONG_REASONS:
            return Result("too_long", audit=audit, detail=detail)
        probe = f"{stderr}\n{text}"
        if not audit["result_seen"] and (NO_SESSION.search(probe) or SESSION_IN_USE.search(probe)):
            return Result("bad_session", audit=audit, detail=detail)
        killed = isinstance(rc, int) and rc < 0  # a signal ended the CLI, so it could not have reported anything
        if (killed or not finished) and not audit["result_seen"]:
            return Result("interrupted", audit=audit, detail=detail)
        return Result("unknown", audit=audit, detail=detail)
    if not text.strip():
        return Result("unknown", audit=audit, detail=detail)
    return Result("ok", text=text, audit=audit)


def neutral_env() -> dict:
    """Only what the CLI needs to find its login: no API key, no project or parent-session variables."""
    env = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "PATH", "LANG", "TMPDIR", "SHELL") if k in os.environ}
    env["DISABLE_AUTOUPDATER"] = "1"
    return env


def process_start(pid: int) -> str:
    """Start time of a live process, "" if there is none: a pid alone can be reused."""
    r = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True)
    return r.stdout.strip()


def attempt_paths(raw_dir: Path, seq: int) -> dict[str, Path]:
    b = Path(raw_dir) / f"call{seq}"
    return {k: Path(f"{b}.{k}") for k in ("lock", "started", "cancelled", "json", "stderr", "done.json", "cmd.json", "prompt",
                                          "child.json", "expired.json", "runner.log")}


def lock_holders(lock: Path) -> set[int]:
    """Who has this attempt's lock file open. The file is unique to the attempt, so this asks the OS directly."""
    r = subprocess.run(["lsof", "-t", str(lock)], capture_output=True, text=True)
    return {int(x) for x in r.stdout.split() if x.strip().isdigit()} - {os.getpid()}


def process_group(pid: int) -> int | None:
    r = subprocess.run(["ps", "-o", "pgid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(r) if r.isdigit() else None


class ClaudeAgent:
    def __init__(self, cwd: Path, effort: str):
        self.cwd, self.effort = Path(cwd), effort
        self.runner: subprocess.Popen | None = None
        self.runner_seq: int | None = None

    def command(self, session: str, new: bool) -> list[str]:
        return [shutil.which("claude") or "claude", "-p", "--model", C.MODEL_ID, "--effort", self.effort,
                "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
                "--system-prompt", C.SYSTEM_PROMPT, "--output-format", "json", "--verbose",
                *(("--session-id", session) if new else ("--resume", session))]

    def write_prompt(self, source: Path, destination: Path) -> None:
        shutil.copy(source, destination)

    def spawn(self, prompt_file: Path, session: str, new: bool, raw_dir: Path, seq: int) -> dict:
        p = attempt_paths(raw_dir, seq)
        p["lock"].touch()
        self.write_prompt(prompt_file, p["prompt"])
        p["cmd.json"].write_text(json.dumps(self.command(session, new)))
        self.runner, self.runner_seq = None, seq
        with open(p["runner.log"], "wb") as log:
            try:
                self.runner = subprocess.Popen([sys.executable, "-I", str(C.HARNESS / "call_runner.py"), str(raw_dir), str(seq),
                                                str(self.cwd), str(C.CALL_SECONDS)], env=neutral_env(), stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=log, start_new_session=True)
            except OSError as e:  # no process at all (fork or file limits): a launch failure like any other
                log.write(f"could not start the runner: {type(e).__name__}: {e}\n".encode())
        return {}

    def discard_attempt(self, raw_dir: Path, seq: int) -> dict:
        """An attempt that never launched: what is known of why, and then its files go, lock first, so a late runner can't
        start. The pilot left 213,000 of them."""
        p = attempt_paths(raw_dir, seq)
        rc = self.runner.poll() if self.runner is not None and self.runner_seq == seq else None
        log = p["runner.log"].read_text(errors="replace")[-2_000:] if p["runner.log"].exists() else ""
        for key in ("lock", "cancelled", "started", "cmd.json", "prompt", "runner.log", "json", "stderr"):
            p[key].unlink(missing_ok=True)
        return {"runner_exit": rc, "runner_log": log}

    def _lock_free(self, p: dict[str, Path], hold=None) -> bool:
        """True if no process of this attempt is alive. hold runs with the lock taken, so no runner can start meanwhile."""
        if not p["lock"].exists():
            return True
        fd = os.open(p["lock"], os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            if hold:
                hold()
            return True
        finally:
            os.close(fd)

    def _expire(self, p: dict[str, Path], info: dict) -> bool:
        """End an attempt past its deadline whose runner did not. Only a process that has this attempt's lock file open
        right now, and is on record (same pid and start time) or in the runner's group, is killed: a reused pid is neither.
        False if something else holds the lock, which a human must look at."""
        known = {(info.get("pid"), info.get("start"))}
        if p["child.json"].exists():
            child = json.loads(p["child.json"].read_text())
            known.add((child.get("pid"), child.get("start")))
        holders = lock_holders(p["lock"])
        targets = []
        for pid in holders:
            if (pid, process_start(pid)) in known or process_group(pid) == info.get("pid"):
                targets.append(pid)
            else:
                return False
        if not targets:
            return False
        p["expired.json"].write_text(json.dumps({"killed": sorted(targets), "t": time.time()}))
        for pid in targets:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return True

    def await_attempt(self, raw_dir: Path, seq: int, sleep=time.sleep, until: float = float("inf"), now=time.time) -> str:
        """Wait for an attempt, this driver's or an earlier one's, but not past `until`, the driver's own wall limit:
          "finished"      the prompt went out and the attempt is over: collect it;
          "not_launched"  it never went out, and now it can't;
          "running"       still going at the wall limit: the attempt stays, and the next driver picks it up;
          "stuck"         past its deadline, held by something that can't be identified: for a human."""
        p = attempt_paths(raw_dir, seq)

        def started() -> dict | None:
            """The runner's record. It is written after the file is created and before the CLI exists, so a file that can't
            be read means the runner died in between: the attempt did not start."""
            try:
                info = json.loads(p["started"].read_text())
            except (OSError, ValueError):
                return None
            return info if isinstance(info, dict) and isinstance(info.get("deadline"), (int, float)) else None

        def cancel_if_not_started() -> None:
            if started() is None:
                p["cancelled"].touch()

        while True:
            own_runner = self.runner is not None and self.runner_seq == seq and self.runner.poll() is None
            if not own_runner and self._lock_free(p, cancel_if_not_started):
                return "finished" if started() is not None else "not_launched"
            info = started()
            if info is not None:
                if now() > info["deadline"] + C.CALL_GRACE_SECONDS and not p["expired.json"].exists():
                    if not self._expire(p, info):
                        return "stuck"
            if now() >= until:
                return "running"
            sleep(min(1.0, max(until - now(), 0.01)))

    def collect(self, raw_dir: Path, seq: int, now: float) -> Result:
        p = attempt_paths(raw_dir, seq)
        out = p["json"].read_text(errors="replace") if p["json"].exists() else ""
        err = p["stderr"].read_text(errors="replace") if p["stderr"].exists() else ""
        done = json.loads(p["done.json"].read_text()) if p["done.json"].exists() else None
        return classify(parse_events(out), err, done["rc"] if done else None, now, finished=done is not None,
                        timed_out=bool(done and done.get("timed_out")) or p["expired.json"].exists())


def scripted_events(session: str, text: str) -> list[dict]:
    return [{"type": "system", "subtype": "init", "session_id": session, "tools": [], "mcp_servers": [], "model": "script"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
            {"type": "result", "subtype": "success", "is_error": False, "result": text, "session_id": session,
             "stop_reason": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0}}]


class ScriptedAgent:
    """A stub with no model calls: reply i of the script answers turn i + 1, so a resumed stub stays in step.
    A callable entry gets the prompt."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[str] = []

    def spawn(self, prompt_file: Path, session: str, new: bool, raw_dir: Path, seq: int) -> dict:
        prompt = Path(prompt_file).read_text()
        self.calls.append(prompt)
        r = self.replies[int(Path(raw_dir).parent.name[1:]) - 1]
        text = r(prompt) if callable(r) else r
        (raw_dir / f"call{seq}.json").write_text(json.dumps(scripted_events(session, text)))
        (raw_dir / f"call{seq}.stderr").write_text("")
        return {}

    def await_attempt(self, raw_dir: Path, seq: int, sleep=time.sleep, until: float = float("inf"), now=time.time) -> str:
        return "finished" if (Path(raw_dir) / f"call{seq}.json").exists() else "not_launched"

    def discard_attempt(self, raw_dir: Path, seq: int) -> dict:
        return {"runner_exit": None, "runner_log": ""}

    def collect(self, raw_dir: Path, seq: int, now: float) -> Result:
        return classify(parse_events((Path(raw_dir) / f"call{seq}.json").read_text()), "", 0, now)
