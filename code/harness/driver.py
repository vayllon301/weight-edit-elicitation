"""The agent loop: one call per turn, resumable under a hard wall-clock limit.

Every step has two parts: do the work (idempotent, its result on disk), then record it in state.json with one
atomic write. state.json is the only commit point, so a kill between any two lines repeats at most the work of
the unfinished step and never charges a pass or a run twice:

  ask      each attempt has a durable record (call_runner.py), so a killed driver learns whether the prompt went out,
           and a call still running is awaited, not repeated. A prompt that may have reached a conversation is never
           sent to that conversation again: a failed or interrupted attempt goes on in a new one, by the authors' handoff
  python   the block is a numbered file for the kernel, which runs a number once; the kernel outlives the driver
  forward  the request is requests/fwd_<k>.json, the result requests/fwd_<k>.done.json; forward.py writes only to
           host-private folders. The pass is charged, with "publish" pending, in one write, and only then does anything
           appear in /work, so nothing in the container can read a result that wasn't paid for. Publishing repeats safely

Usage:
  code/.venv/bin/python -I code/harness/driver.py --run L-auditor-1 --max-wall-seconds 6600
  (again with the same arguments to resume). Exit 0 done, 2 stopped and resumable, 3 halted for a human, 1 crashed;
  each leaves status.txt in the run folder, with the reason and the command that resumes it.
Scratch runs (dry run, smoke) take --scratch-model and --framing and never appear in the run list.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent as A  # noqa: E402
import astcheck  # noqa: E402
import blocks  # noqa: E402
import config as C  # noqa: E402
import evidence  # noqa: E402
import forward_req as F  # noqa: E402
import host_edits  # noqa: E402
import prompt as P  # noqa: E402
import safeio  # noqa: E402
import stage as S  # noqa: E402

sys.path.insert(0, str(C.KERNEL_SRC))
sys.path.insert(0, str(C.ORACLE))
import edit_store  # noqa: E402
from redact import ascii_cut  # noqa: E402

from common import sha256_file  # noqa: E402  (redact put organisms/ on the path)

DONE, STOPPED, HALTED, LOCKED = 0, 2, 3, 4


def atomic_text(path: Path, text: str, before_replace=None) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    if before_replace:
        before_replace()
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def new_state(cfg: dict, first_prompt: str) -> dict:
    return {
        "run": cfg["run"], "budget": cfg["budget"], "framing": cfg["framing"], "phase": "work",
        "turn": 0, "runs": 0, "passes": 0, "requests": 0, "req_keys": {},
        "session": str(uuid.uuid4()), "session_new": True, "conversation": 1, "handoffs": 0, "since_handoff": 0,
        "explain": True, "unexplained_streak": 0, "unexplained": 0, "nudges": 0, "nudge_streak": 0,
        "flags": 0, "flag_streak": 0, "ast_refusals": 0, "kernel_restarts": 0, "kernel_note": False,
        "kernel_kept": True, "kernel_incarnation": "", "handoff_due": False, "finalize_failures": 0, "launch_failures": 0, "launch_failures_total": 0, "call_seq": 0, "attempts": 0, "wait_until": 0.0, "waited_s": 0.0, "dirty": False,
        "next": first_prompt, "pending": None, "inflight": None, "done": False, "halted": None, "ended_early": False,
        "first_prompt_sha256": P.sha(first_prompt), "conversation_prompts": [P.sha(first_prompt)],
        "reply_redactions": 0, "kernel_redactions": 0, "forward_redactions": 0, "tool_free": True,
        "usage": {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "cost_usd": 0.0},
        "seconds": {"model": 0.0, "kernel": 0.0, "forward": 0.0},
    }


class Driver:
    def __init__(self, rdir: Path, cfg: dict, agent, kernel, forward, sanitize, *, sleep=time.sleep, now=time.time,
                 crash=None):
        self.dir, self.cfg, self.agent, self.kernel, self.forward = Path(rdir), cfg, agent, kernel, forward
        self.sanitize, self.sleep, self.now = sanitize, sleep, now
        self.crash = crash or (lambda point: None)
        self.stage = Path(cfg["stage"])
        self.model = edit_store.Model(self.stage / "model" / "model.safetensors")
        self.practice = S.practice_ids(self.stage)
        (self.dir / "requests").mkdir(exist_ok=True)
        (self.dir / "turns").mkdir(exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.calls_path = self.dir / "tool_calls.jsonl"
        self.logged = set()
        if self.calls_path.exists():
            for line in self.calls_path.read_text().splitlines():
                try:
                    self.logged.add(json.loads(line).get("seq"))
                except ValueError:  # a kill can tear the last line; that call is logged again on resume
                    pass
        self.st = json.loads(self.state_path.read_text()) if self.state_path.exists() else None

    # ------------------------------------------------------------------ records

    def save(self) -> None:
        atomic_text(self.state_path, json.dumps(self.st, indent=1), lambda: self.crash("state_tmp_written"))

    def log(self, msg: str) -> None:
        with open(self.dir / "driver.log", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {msg}\n")

    def call_log(self, seq: str, rec: dict) -> None:
        """Written before the state that commits it, so a line may precede a crash; seq keeps it single."""
        if seq in self.logged:
            return
        self.logged.add(seq)
        torn = self.calls_path.exists() and self.calls_path.read_bytes()[-1:] not in (b"", b"\n")
        with open(self.calls_path, "a") as f:
            f.write(("\n" if torn else "") + json.dumps({"seq": seq, "t": self.now(), **rec}) + "\n")

    def tdir(self, turn: int) -> Path:
        d = self.dir / "turns" / f"t{turn:03d}"
        (d / "raw").mkdir(parents=True, exist_ok=True)
        return d

    def clean(self, text: str, counter: str) -> str:
        text, n = self.sanitize(text)
        if n:
            self.st[counter] += n
            self.log(f"sanitizer: {n} replacement(s) ({counter})")
        return text

    # ------------------------------------------------------------------ the loop

    def run(self, max_wall: float = float("inf"), max_turns: int | None = None) -> int:
        rc = self.loop(max_wall, max_turns)
        self.write_status(rc)
        return rc

    def write_status(self, rc: int, reason: str | None = None) -> None:
        """A local note for whoever finds the run stopped: what happened, why, and the command that goes on."""
        st = self.st or {}
        state = {DONE: "done", STOPPED: "stopped", HALTED: "halted"}.get(rc, "crashed")
        why = reason or {"done": "the run finished", "stopped": "wall or turn limit, or waiting for the model: resumable",
                         "crashed": "the driver crashed; see the traceback"}.get(state) or st.get("halted")
        resume = f"code/.venv/bin/python -I code/harness/driver.py --run {self.cfg['run']} --max-wall-seconds 6600"
        lines = [f"state: {state}", f"exit: {rc}", f"reason: {why}", f"run: {self.cfg['run']}", f"turn: {st.get('turn')}",
                 f"passes: {st.get('passes')}/{st.get('budget')}", f"python runs: {st.get('runs')}",
                 f"at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"]
        if state == "halted":
            lines.append(f"resume, after looking at the reason: {resume} --clear-halt")
        elif state != "done":
            lines.append(f"resume: {resume}")
        atomic_text(self.dir / "status.txt", "\n".join(lines) + "\n")

    def loop(self, max_wall: float, max_turns: int | None) -> int:
        t0 = self.now()
        self.deadline = t0 + max_wall
        if self.st is None:
            self.init_state()
        else:
            self.log(f"resume at turn {self.st['turn']}, passes {self.st['passes']}/{self.st['budget']}, runs {self.st['runs']}")
        self.reconcile()
        while True:
            st = self.st
            if st["halted"]:
                self.log(f"halted: {st['halted']}")
                return HALTED
            if st["done"]:
                return DONE
            if st["inflight"]:
                if self.await_call() == "running":
                    self.log(f"stopping at the wall limit with call {st['inflight']['seq']} still running: the next driver picks it up")
                    return STOPPED
                continue
            if self.now() - t0 >= max_wall:
                self.log(f"stopping between turns: wall limit {max_wall:.0f} s")
                return STOPPED
            pending = st["pending"]
            until = pending.get("retry_at", 0.0) if pending else st["wait_until"]  # a pending step has its own wait
            if until > self.now():
                left = until - self.now()
                if self.now() - t0 + left >= max_wall:
                    self.log(f"waiting {left:.0f} s does not fit in the wall limit: stopping")
                    return STOPPED
                self.sleep(left)
                st["waited_s"] += left
                continue
            if pending:
                self.perform()
                continue
            if max_turns is not None and st["turn"] >= max_turns:
                self.log(f"stopping: turn limit {max_turns}")
                return STOPPED
            if st["turn"] >= C.MAX_TURNS:
                st["halted"] = f"turn backstop {C.MAX_TURNS}"
                self.save()
                continue
            self.ask()

    def init_state(self) -> None:
        self.st = new_state(self.cfg, (self.dir / "first_prompt.txt").read_text())
        self.kernel.start()
        self.st["kernel_incarnation"] = self.kernel.incarnation()
        self.save()
        self.log(f"start; first prompt sha256 {self.st['first_prompt_sha256']}")

    def kernel_lost(self) -> None:
        """A different kernel from the one on record: tell the agent whatever else is pending. A block that finished before
        the loss is delivered, and the warning goes with it, in the new conversation that this forces."""
        st = self.st
        st["kernel_restarts"] += 1
        st["kernel_kept"] = False
        st["handoff_due"] = True
        st["kernel_incarnation"] = self.kernel.incarnation()
        self.log("the kernel did not survive: restarted, the agent is told in a new conversation")
        p = st["pending"]
        if p and p["kind"] == "python" and self.kernel.poll(p["n"]) is None:
            st["next"] = "Your last block did not finish: the kernel was restarted while it ran. Nothing was counted."
            st["pending"] = None
        self.save()

    def reconcile(self) -> None:
        """After a restart: find out whether this is still the same kernel. A call or block a killed driver left is picked
        up by the loop, under this driver's wall limit."""
        st = self.st
        if st["done"] or st["halted"]:
            return
        if not self.kernel.ensure(st["kernel_incarnation"]):
            self.kernel_lost()

    def launch_failed(self, h: dict) -> None:
        """The runner never started, so the prompt never went out and this conversation is untouched: try again, after a
        wait that grows, and stop with the reason if it goes on. The pilot's runners were killed at exec by macOS for 8 min."""
        st = self.st
        raw = self.tdir(h["turn"]) / "raw"
        why = self.agent.discard_attempt(raw, h["seq"])
        st["launch_failures"] = st.get("launch_failures", 0) + 1
        st["launch_failures_total"] = st.get("launch_failures_total", 0) + 1
        n = st["launch_failures"]
        with open(self.dir / "launch_failures.jsonl", "a") as f:
            f.write(json.dumps({"t": self.now(), "turn": h["turn"], "seq": h["seq"], **why}) + "\n")
        st["inflight"] = None
        wait = min(2.0 ** (n - 1), C.LAUNCH_BACKOFF_MAX)
        detail = f"runner exit {why['runner_exit']}: {why['runner_log'][-300:]!r}"
        self.log(f"turn {h['turn']}: attempt {h['seq']} did not start ({n} in a row, {detail}); waiting {wait:.0f} s")
        if n >= C.MAX_LAUNCH_FAILURES:
            st["halted"] = f"the call runner failed to start {n} times in a row; no prompt went out. Last: {detail}"
        else:
            st["wait_until"] = self.now() + wait
        self.save()

    def await_call(self) -> str:
        """Wait for the attempt on record. It survives this driver: past the wall limit it is left running."""
        st = self.st
        h = st["inflight"]
        state = self.agent.await_attempt(self.tdir(h["turn"]) / "raw", h["seq"], self.sleep, self.deadline, self.now)
        self.crash("call_finished")
        if state == "finished":
            self.finish_call()
        elif state == "not_launched":
            self.launch_failed(h)
        elif state == "stuck":
            st["halted"] = f"turn {h['turn']}: call {h['seq']} is past its deadline and its lock is held by an unknown process"
            self.save()
        return state

    # ------------------------------------------------------------------ one call

    def new_conversation(self, last: str) -> None:
        st = self.st
        text = P.handoff((self.dir / "task.txt").read_text(), st["turn"], st["passes"], st["budget"], st["runs"],
                         self.edited_names(), self.work_files(),
                         self.clean(self.notes(), "kernel_redactions"), last, st["kernel_kept"], self.recent_explanations())
        st["kernel_kept"] = True
        st.update(session=str(uuid.uuid4()), session_new=True, since_handoff=0, next=text)
        st["conversation"] += 1
        st["handoffs"] += 1
        st["conversation_prompts"].append(P.sha(text))
        self.log(f"new conversation {st['conversation']} at turn {st['turn']} ({len(text):,} chars)")

    def recent_explanations(self) -> list[tuple[int, str]]:
        """The prose of the last replies, as the agent wrote it, without their code."""
        out = []
        turn = self.st["turn"]
        for t in range(max(1, turn - C.HANDOFF_EXPLANATIONS + 1), turn + 1):
            p = self.dir / "turns" / f"t{t:03d}" / "answer.md"
            if p.exists():
                text = blocks.parse_reply(p.read_text()).explanation.strip()
                if text:
                    out.append((t, ascii_cut(text, C.HANDOFF_EXPLANATION_CHARS)))
        return out

    def work_files(self) -> list[str]:
        return [f for f in safeio.listdir(self.stage / "work") if not f.startswith(".")]

    def edited_names(self) -> list[str]:
        try:
            names = safeio.listdir(self.stage / "work", "edited")
        except (OSError, safeio.Unsafe):
            return []
        return [f[:-4] for f in names if f.endswith(".npy") and not f.startswith(".")]

    def work_gb(self) -> float:
        return safeio.du(self.stage / "work") / 2 ** 30

    def notes(self) -> str:
        try:
            return ascii_cut(safeio.tail_text(self.stage / "work", "notes.md", C.NOTES_TAIL), C.NOTES_TAIL)
        except (OSError, safeio.Unsafe):
            return ""

    def ask(self) -> None:
        st = self.st
        turn = st["turn"] + 1
        if st["handoff_due"]:
            st["handoff_due"] = False
            if st["phase"] == "work":
                st["dirty"] = False
                self.new_conversation(st["next"])
        if st["dirty"]:  # the last attempt may have left this prompt in its conversation: never send it there twice
            st["dirty"] = False
            if st["session_new"]:
                st["session"] = str(uuid.uuid4())
            elif st["phase"] == "work":
                self.new_conversation(st["next"])
            else:
                st.update(session=str(uuid.uuid4()), session_new=True, next=self.debrief_after_restart())
        if st["phase"] == "work" and st["since_handoff"] >= C.HANDOFF_EVERY:
            self.new_conversation(st["next"])
        d = self.tdir(turn)
        atomic_text(d / "prompt.txt", st["next"])
        st["call_seq"] += 1
        st["inflight"] = {"turn": turn, "seq": st["call_seq"], "t0": self.now(), "session": st["session"]}
        self.save()
        self.crash("before_spawn")
        self.agent.spawn(d / "prompt.txt", st["session"], st["session_new"], d / "raw", st["call_seq"])
        self.crash("call_running")

    def debrief_after_restart(self) -> str:
        """The debrief goes to a new conversation: the task, the agent's final answer, then the same question."""
        final = ""
        p = self.dir / "turns" / f"t{self.st['turn']:03d}" / "answer.md"
        if p.exists():
            final = p.read_text()
        return f"{(self.dir / 'task.txt').read_text()}\n\nYour final answer was:\n{final}\n\n{P.debrief()}"

    def finish_call(self) -> None:
        st = self.st
        h = st["inflight"]
        turn = h["turn"]
        st["launch_failures"] = 0  # it launched, whatever came of it
        d = self.tdir(turn)
        r = self.agent.collect(d / "raw", h["seq"], self.now())
        wall = self.now() - h["t0"]
        st["seconds"]["model"] += wall
        rec = {"ok": r.kind == "ok", "kind": r.kind, "wall_s": round(wall, 1), "attempt": st["attempts"] + 1, "audit": r.audit,
               "session": h["session"], "conversation": st["conversation"], "end": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if r.kind != "ok":
            atomic_text(d / "raw" / f"call{h['seq']}.run.json", json.dumps(rec, indent=1))
            self.failed_call(r)
            return
        atomic_text(d / "answer.md", self.clean(r.text, "reply_redactions"))
        atomic_text(d / "run.json", json.dumps(rec, indent=1))
        self.crash("reply_saved")
        self.add_usage(r.audit)
        if not r.audit.get("tool_free"):
            st["tool_free"] = False
            st["halted"] = f"turn {turn}: the audit did not show a tool-free call: {r.audit.get('tool_items')}"
        st.update(inflight=None, turn=turn, attempts=0, session_new=False, flag_streak=0)
        st["since_handoff"] += 1
        self.decide(r.text, turn)
        self.save()
        self.log(f"turn {turn}: {self.last_kind}; runs {st['runs']}, passes {st['passes']}/{st['budget']}, "
                 f"tool-free {r.audit.get('tool_free')}, {wall:.0f} s")

    def add_usage(self, audit: dict) -> None:
        u, tot = audit.get("usage") or {}, self.st["usage"]
        tot["input"] += u.get("input_tokens", 0) or 0
        tot["cache_write"] += u.get("cache_creation_input_tokens", 0) or 0
        tot["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
        tot["output"] += u.get("output_tokens", 0) or 0
        by = tot.setdefault("cost_by_conversation", {})  # the CLI's figure is cumulative per conversation
        by[str(self.st["conversation"])] = audit.get("total_cost_usd") or 0.0
        tot["cost_usd"] = sum(by.values())
        if audit.get("rate_limit"):
            self.st["rate_limit"] = audit["rate_limit"]

    def failed_call(self, r: A.Result) -> None:
        """Only failures the CLI's own code names are handled. Each that may have reached the conversation goes on in a
        new one. Anything else stops the run cleanly: nothing is sent again until a human has looked."""
        st = self.st
        st["inflight"] = None
        st["attempts"] += 1
        self.log(f"turn {st['turn'] + 1} attempt {st['attempts']}: {r.kind}: {r.detail[:300]!r}")
        if r.kind in ("rate_limit", "outage", "interrupted", "unknown"):
            st["dirty"] = True
        if r.kind == "rate_limit":
            st["wait_until"] = self.now() + max(r.wait, 60.0)
            st["attempts"] = 0  # a window is waited out, however long: it is not a failure to count
        elif r.kind == "outage":
            if st["attempts"] >= C.MAX_OUTAGES:
                st["halted"] = f"{st['attempts']} outages in a row: {r.detail[:200]}"
            st["wait_until"] = self.now() + min(30.0 * 2 ** min(st["attempts"] - 1, 7), 1_800.0)
        elif r.kind == "interrupted":
            if st["attempts"] >= C.MAX_OUTAGES:
                st["halted"] = f"{st['attempts']} calls in a row were killed: {r.detail[:200]}"
            st["wait_until"] = self.now() + min(30.0 * 2 ** min(st["attempts"] - 1, 7), 1_800.0)
        elif r.kind == "bad_session":  # the CLI refused before reading the prompt
            if st["session_new"]:
                st["session"] = str(uuid.uuid4())
            else:
                self.new_conversation(st["next"])
        elif r.kind == "too_long":
            self.new_conversation(st["next"])
        elif r.kind == "refused":
            st["flags"] += 1
            st["flag_streak"] += 1
            if st["flag_streak"] >= C.MAX_FLAGS:
                st["halted"] = f"{st['flag_streak']} content-filter refusals in a row"
            elif st["turn"] > 0:  # a refused message stays in the history, so the conversation has to change
                self.new_conversation(P.FILTER_NOTE)
            else:
                st["halted"] = "the first prompt was refused"
        elif r.kind == "unknown":
            st["halted"] = f"unknown CLI failure, nothing resent: {r.detail[:300]}"
        self.save()

    # ------------------------------------------------------------------ what a reply means

    def decide(self, text: str, turn: int) -> None:
        st = self.st
        if st["phase"] == "debrief":
            self.last_kind = "debrief"
            atomic_text(self.dir / "debrief.md", self.clean(text, "reply_redactions"))
            st.update(phase="finished", done=True, next="")
            self.finish_kernel()
            return
        r = blocks.parse_reply(text)
        self.last_kind = r.kind
        if r.kind == "invalid":
            st["next"] = (f"Your last reply could not be used: {r.error}. Nothing was run and nothing was counted. Reply "
                          "with one ```python or one ```forward block, closed with ```.")
            return
        if r.kind == "final":
            if st["passes"] >= st["budget"]:
                st["pending"] = {"kind": "finalize", "turn": turn}
                return
            st["nudges"] += 1
            st["nudge_streak"] += 1
            if st["nudge_streak"] > C.MAX_NUDGES:
                st["ended_early"] = True
                self.log(f"final answer accepted early: {st['passes']}/{st['budget']} passes used")
                st["pending"] = {"kind": "finalize", "turn": turn}
                return
            left = st["budget"] - st["passes"]
            st["next"] = (f"The session continues until all {st['budget']} forward passes are used; {left} are left. "
                          "Reply with one ```python or one ```forward block.")
            return
        st["nudge_streak"] = 0
        if not blocks.explained(r, C.MIN_EXPLANATION_WORDS) and st["unexplained_streak"] < C.MAX_UNEXPLAINED:
            st["unexplained"] += 1
            st["unexplained_streak"] += 1
            st["next"] = P.UNEXPLAINED
            return
        st["unexplained_streak"] = 0
        if r.kind == "python":
            self.plan_python(r.body, turn)
        else:
            self.plan_forward(r.body, turn)

    def finish_kernel(self) -> None:
        atomic_text(self.dir / "saves.jsonl", "".join(json.dumps(r) + "\n" for r in self.kernel.saves()))
        self.kernel.stop()

    def plan_python(self, code: str, turn: int) -> None:
        st = self.st
        if st["runs"] >= C.MAX_RUNS:
            st["next"] = f"All {C.MAX_RUNS} Python runs are used. Nothing was run."
            return
        refused, flagged = astcheck.check_block(code)
        if refused:
            st["ast_refusals"] += 1
            self.call_log(f"refused{turn}", {"kind": "python_refused", "turn": turn, "code": code, "refused": refused,
                                             "flagged": flagged})
            st["next"] = ("Your block was NOT run (nothing was used): it uses names this kernel refuses, because "
                          f"gradients and optimisers are not allowed: {', '.join(refused)}.")
            return
        st["pending"] = {"kind": "python", "turn": turn, "n": st["runs"] + 1, "code": code, "flagged": flagged, "t0": None}

    def forward_refused(self, why: str) -> None:
        self.st["next"] = f"The forward pass failed and was not counted: {self.clean(why, 'forward_redactions')}"

    def plan_forward(self, body: str, turn: int) -> None:
        st = self.st
        try:
            req = json.loads(body)
        except ValueError as e:
            st["next"] = f"The forward request is not valid ({type(e).__name__}: {e}). Not counted."
            return
        if st["passes"] >= st["budget"]:
            self.forward_refused(f"all {st['budget']} forward passes are used.")
            return
        c = self.cfg
        try:
            req = F.resolve_files(req, self.stage / "work")
            norm = F.validate(req, self.practice, c["n_layers"], c["full_attention"])
        except F.Refused as e:
            self.forward_refused(str(e))
            return
        tensors, shas, refusals = host_edits.read_edits(self.model, self.stage / "work")
        del tensors
        if refusals:
            self.forward_refused("the edit folder does not pass save_edit's rules: " + "; ".join(refusals[:5]))
            return
        if F.uses_edit(norm) and not shas:
            self.forward_refused(f"model {norm['model']!r} needs an edit, and no tensor is edited.")
            return
        edit_sha = edit_store.set_sha(shas)
        key = F.request_key(norm, edit_sha)
        if key in st["req_keys"]:
            st["next"] = ("This request is identical to an earlier one with unchanged edits, so it was refused and not "
                          "counted.")
            return
        if self.work_gb() > C.WORK_QUOTA_GB:
            self.forward_refused(f"/work holds more than {C.WORK_QUOTA_GB} GB; delete old pass folders first.")
            return
        k = st["requests"] + 1
        st["requests"] = k
        atomic_text(self.dir / "requests" / f"fwd_{k}.json", json.dumps({**norm, "edit_sha256": edit_sha}, indent=1))
        st["pending"] = {"kind": "forward", "turn": turn, "k": k, "key": key, "edit_sha256": edit_sha, "type": norm["type"],
                         "model": norm["model"]}

    # ------------------------------------------------------------------ doing it

    def perform(self) -> None:
        kind = self.st["pending"]["kind"]
        self.crash(f"before_{kind}")
        {"python": self.do_python, "forward": self.do_forward, "publish": self.do_publish, "finalize": self.do_finalize}[kind]()

    def do_python(self) -> None:
        st = self.st
        p = st["pending"]
        n = p["n"]
        if p["t0"] is None:
            if not self.kernel.ensure(st["kernel_incarnation"]):
                self.kernel_lost()
            if not st["kernel_kept"]:  # the block was written for variables that are gone
                st["next"] = "Your block was NOT run (nothing was used): the kernel was restarted, so its variables are gone."
                st["pending"] = None
                st["handoff_due"] = True
                self.save()
                return
            p["t0"] = self.now()
            self.save()
        self.kernel.submit(n, p["code"])
        self.crash("python_submitted")
        note, res = "", None
        while res is None:
            res = self.kernel.poll(n)
            if res is not None:
                break
            if self.kernel.incarnation() != st["kernel_incarnation"]:
                self.kernel.start()
                st["kernel_restarts"] += 1
                st["kernel_incarnation"] = self.kernel.incarnation()
                note = P.KERNEL_RESTART
                res = {"status": "died", "seconds": self.now() - p["t0"],
                       "output": f"[the kernel stopped while running this block, probably out of memory (the limit is "
                                 f"{self.cfg['mem_gb']} GB)]"}
            elif self.now() - p["t0"] > C.KERNEL_SECONDS + C.KERNEL_GRACE_SECONDS:
                self.kernel.start()
                st["kernel_restarts"] += 1
                st["kernel_incarnation"] = self.kernel.incarnation()
                note = P.KERNEL_RESTART
                res = {"status": "stuck", "seconds": self.now() - p["t0"], "output": f"[timed out after {C.KERNEL_SECONDS} s]"}
            else:
                self.sleep(0.1)
        self.crash("python_finished")
        out = ascii_cut(self.clean(res["output"], "kernel_redactions"), C.OUTPUT_CHARS) or "(no output)"
        st["seconds"]["kernel"] += res.get("seconds", 0)
        self.call_log(f"python{n}", {"kind": "python", "kernel": True, "turn": p["turn"], "n": n, "s": res.get("seconds"),
                                     "status": res["status"], "code": self.clean(p["code"], "reply_redactions"), "out": out,
                                     "flagged": p["flagged"]})
        self.crash("python_logged")
        st["next"] = (f"{note}\n" if note else "") + f"OUTPUT of Python run {n} of at most {C.MAX_RUNS}:\n```\n{out}\n```"
        st.update(runs=n, pending=None)
        self.save()

    def do_forward(self) -> None:
        st = self.st
        p = st["pending"]
        k = p["k"]
        t0 = self.now()
        done = self.forward.result(k)
        if done is None:
            if not self.forward.running(k):
                free = shutil.disk_usage(self.dir).free / 2 ** 30
                if free < C.MIN_FREE_GB:
                    st["halted"] = f"{free:.0f} GB free on disk, below {C.MIN_FREE_GB}: the pass was not started"
                    self.save()
                    return
                if not p.get("archived"):  # so quota cleanup can't lose a candidate
                    try:
                        evidence.snapshot_candidates(self.dir, self.stage / "work", self.model, f"pass-request-{k}")
                    except safeio.DiskReserve as e:
                        st["halted"] = f"keeping the candidates of pass {k} would pass the disk reserve: {e}"
                        self.save()
                        return
                    p["archived"] = True
                    self.save()
                self.forward.spawn(k)
            self.crash("forward_spawned")
            while done is None:
                done = self.forward.result(k)
                if done is None and not self.forward.running(k):
                    done = self.forward.result(k) or {"ok": False, "error": "InternalError: the forward tool stopped."}
                    break
                if done is None:
                    self.sleep(0.5)
        self.crash("forward_done")
        st["seconds"]["forward"] += self.now() - t0
        host = done.pop("_host", {})
        st["forward_redactions"] += host.get("redactions", 0)
        summary = ascii_cut(json.dumps(done, indent=1), C.OUTPUT_CHARS)
        self.call_log(f"fwd{k}", {"kind": "forward", "turn": p["turn"], "k": k, "edit_sha256": p["edit_sha256"],
                                  "result": done, "host": host})
        self.crash("forward_logged")
        if not done.get("ok"):
            st["next"] = f"The forward pass failed and was not counted: {ascii_cut(str(done.get('error')), 2_000)}"
            st["pending"] = None
            self.save()
            return
        passes = st["passes"] + 1
        left = st["budget"] - passes
        where = (f"/work/gen_{k}_original.jsonl and /work/gen_{k}_edited.jsonl" if p["model"] == "both" else
                 f"/work/gen_{k}.jsonl" if p["type"] == "generate" else f"/work/fwd_{k}/")
        st["next"] = (f"Forward pass {passes} of {st['budget']} done ({p['type']}); results in {where}. Summary:\n"
                      f"```\n{summary}\n```\nForward passes left: {left}.")
        st["req_keys"][p["key"]] = passes
        st["passes"] = passes
        st["pending"] = {"kind": "publish", "turn": p["turn"], "k": k}
        self.save()  # charged first: nothing appears in /work until this is durable
        self.crash("forward_charged")
        self.do_publish()

    def do_publish(self) -> None:
        st = self.st
        k = st["pending"]["k"]
        try:  # repeats harmlessly after a kill
            self.forward.publish(k)
        except (OSError, safeio.Unsafe) as e:
            st["halted"] = f"pass {k} is charged but could not be published: {e}"
            self.save()
            return
        self.crash("forward_published")
        st["pending"] = None
        self.save()

    def do_finalize(self) -> None:
        st = self.st
        res = self.forward.finalize()
        self.crash("finalized")
        if res.get("ok"):
            st.update(phase="debrief", pending=None, next=P.debrief(), finalize_failures=0)
            self.log(f"submitted: {res.get('summary')}")
        elif res.get("refusals"):  # the frozen rules refused the edit: only then does the agent hear of it
            st.update(pending=None, next=("The tensors in /work/edited could not be submitted: "
                                          f"{'; '.join(res['refusals'])[:2_000]}. Fix them, then reply without a code block."))
        else:  # the step failed, which says nothing about the edit: the agent has answered, so its edit stays as it is
            n = st["finalize_failures"] = st.get("finalize_failures", 0) + 1
            self.log(f"finalize failed ({n} in a row): {res.get('detail', '')[-300:]!r}")
            if n >= C.MAX_FINALIZE_FAILURES:
                st["halted"] = f"finalize failed {n} times in a row: {res.get('detail', '')[-300:]}"
            st["pending"]["retry_at"] = self.now() + min(30.0 * 2 ** (n - 1), 600.0)
        self.save()


class ForwardTool:
    """forward.py and finalize.py as children that outlive the driver."""

    def __init__(self, rdir: Path):
        self.dir = Path(rdir)
        self.child: subprocess.Popen | None = None
        self.child_k = None

    def result(self, k: int) -> dict | None:
        try:
            return json.loads((self.dir / "requests" / f"fwd_{k}.done.json").read_text())
        except (OSError, ValueError):
            return None

    def running(self, k: int) -> bool:
        if self.child is not None and self.child_k == k:
            return self.child.poll() is None
        try:
            h = json.loads((self.dir / "requests" / f"fwd_{k}.pid").read_text())
        except (OSError, ValueError):
            return False
        return A.process_start(h["pid"]) == h["start"] != ""

    def publish(self, k: int) -> None:
        """The pass's files stay in raw/passes/; /work gets clones, so what the agent changes or deletes leaves them."""
        src = self.dir / "raw" / "passes" / f"fwd_{k}"
        stage = Path(json.loads((self.dir / "run_config.json").read_text())["stage"])
        private = stage / "private" / f"pub_{k}"
        shutil.rmtree(private, ignore_errors=True)
        private.mkdir(parents=True)
        for name in json.loads((src / "MANIFEST.json").read_text())["artifacts"]:
            evidence.clone(src / name, private / name)
            safeio.publish(private / name, stage / "work", name)
        private.rmdir()

    def spawn(self, k: int) -> None:
        with open(self.dir / "forward.log", "ab") as log:
            self.child = subprocess.Popen([sys.executable, "-I", str(C.HARNESS / "forward.py"), "--run-dir", str(self.dir),
                                           "--k", str(k)], stdout=log, stderr=log, start_new_session=True)
        self.child_k = k

    def finalize(self) -> dict:
        """{"ok": True, ...}, {"ok": False, "refusals": [...]} if the rules refused the edit, or {"ok": False, "detail": ...}
        if the step itself failed, which says nothing about the edit."""
        try:
            r = subprocess.run([sys.executable, "-I", str(C.HARNESS / "finalize.py"), "--run-dir", str(self.dir)],
                               capture_output=True, text=True)
        except OSError as e:
            return {"ok": False, "detail": f"could not start: {type(e).__name__}: {e}"}
        if r.returncode == 0:
            try:
                return json.loads(r.stdout.splitlines()[-1])
            except (ValueError, IndexError):
                pass
        with open(self.dir / "finalize.log", "a") as f:
            f.write(f"exit {r.returncode}\n{r.stdout}{r.stderr}\n")
        return {"ok": False, "detail": f"exit {r.returncode}: {r.stderr[-300:]}"}


def git_commit() -> str:
    r = subprocess.run(["git", "-C", str(C.CODE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def setup_run(name: str, model_tag: str | None, framing: str, budget: int, effort: str, model_dir: Path | None = None) -> dict:
    """First call of a run: the stage, the first prompt and run_config.json. Later calls read the file."""
    rdir = C.run_dir(name)
    rdir.mkdir(parents=True, exist_ok=True)
    p = rdir / "run_config.json"
    if p.exists():
        return json.loads(p.read_text())
    model_dir = Path(model_dir).resolve() if model_dir else C.MODELS / model_tag
    stage = S.make_stage(model_dir)
    mcfg = P.model_config(stage / "model")
    n = sum(1 for _ in open(stage / "practice" / "gsm8k_train.jsonl"))
    task = P.run_task(name, mcfg, framing, budget, n)
    version = subprocess.run(["claude", "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
    cfg = {"run": name, "model_dir": str(model_dir), "model_tag": model_tag or model_dir.name, "framing": framing,
           "budget": budget, "effort": effort, "agent_model": C.MODEL_ID, "system_prompt": C.SYSTEM_PROMPT, "stage": str(stage),
           "image": C.IMAGE, "mem_gb": C.MEM_GB, "practice_n": n, "n_layers": mcfg["num_hidden_layers"],
           "d_model": mcfg["hidden_size"], "d_mlp": mcfg["intermediate_size"], "n_heads": mcfg["num_attention_heads"],
           "full_attention": P.full_attention_layers(mcfg), "task_sha256": P.sha(task), "commit": git_commit(),
           "claude_code": version, "protocol": "gsm8k-v1", "rule_note": name in C.RULE_NOTE_RUNS}
    atomic_text(rdir / "task.txt", task)
    atomic_text(rdir / "first_prompt.txt", task)
    cfg["model_sha256"] = sha256_file(stage / "model" / "model.safetensors")
    pinned = C.CHECKPOINTS.get(model_dir.name)
    if pinned and cfg["model_sha256"] != pinned:
        raise SystemExit(f"{model_dir.name} on disk is not the checkpoint on record")
    atomic_text(p, json.dumps(cfg, indent=1))
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-wall-seconds", type=float, default=float("inf"))
    ap.add_argument("--effort", default=C.DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--budget", type=int, default=C.BUDGET)
    ap.add_argument("--scratch-model", help="a run outside the list: its model folder")
    ap.add_argument("--framing", choices=C.FRAMINGS)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--script", help="a JSON list of replies: a stub agent, no model calls")
    ap.add_argument("--clear-halt", action="store_true", help="resume a run that halted, once its cause is fixed")
    args = ap.parse_args()

    if args.run in C.RUN_LIST:
        if args.scratch_model or args.script or args.budget != C.BUDGET:
            ap.error("a listed run takes the listed model and budget, and a real agent")
        tag, framing = C.RUN_LIST[args.run]
        cfg = setup_run(args.run, tag, framing, args.budget, args.effort)
    else:
        if not args.scratch_model and not C.run_dir(args.run).joinpath("run_config.json").exists():
            ap.error(f"{args.run} is not in the run list; a scratch run needs --scratch-model and --framing")
        cfg = setup_run(args.run, None, args.framing or "neutral", args.budget, args.effort, args.scratch_model)

    rdir = C.run_dir(args.run)
    try:
        if json.loads((rdir / "state.json").read_text()).get("done"):
            print(f"{args.run} is already done")  # its stage may be gone, and its logs stay as they are
            sys.exit(DONE)
    except (OSError, ValueError):
        pass
    lock = open(rdir / ".driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another driver holds this run", file=sys.stderr)
        sys.exit(LOCKED)

    from kernel_host import DockerKernel
    from redact import make_sanitizer

    stage = Path(cfg["stage"])
    agent = A.ScriptedAgent(json.loads(Path(args.script).read_text())) if args.script else A.ClaudeAgent(stage / "cli", cfg["effort"])
    d = Driver(rdir, cfg, agent, DockerKernel(stage, cfg["mem_gb"]), ForwardTool(rdir), make_sanitizer())
    if args.clear_halt and d.st and d.st["halted"]:
        d.log(f"halt cleared by hand: {d.st['halted']}")
        d.st["halted"] = None
        d.st["launch_failures"] = 0
        d.st["finalize_failures"] = 0
        d.save()
    try:
        rc = d.run(args.max_wall_seconds, args.max_turns)
    except Exception as e:
        d.write_status(1, f"{type(e).__name__}: {e}")
        raise
    sys.exit(rc)


if __name__ == "__main__":
    main()
