# Agent harness (M3)

Runs the design's six agent runs: Claude Opus 5.5, driven turn by turn through `claude -p`, with a Python kernel over a
model's weights, a forward-pass tool and a budget of 50 passes. Built to the study design, `the study design` section 5
and `docs/EDIT_RULES.md` sections 4-5. `edit_rules.py` is frozen: the kernel gets the same code with its docstrings
removed (a test compares the ASTs), and the host checks every edit again.

The prompts, the handoff text and the messages are adapted from Takehi, Teoh, Zhang and Khattab, "Sparks of Weight-Space
Programming": <https://github.com/RikiyaT/lm_understand-logs> (logs CC BY 4.0, scripts MIT).

## Run

    docker build -t edit-kernel:1 code/harness/docker
    code/.venv/bin/python -I code/harness/driver.py --run L-auditor-1 --max-wall-seconds 6600   # again to resume
    code/harness/queue.sh 6600                     # L-neutral-1 only (the design: review it before the other five)
    QUEUE_REVIEWED=1 code/harness/queue.sh 6600    # all six, in order
    code/harness/run_all.sh 6600 [total_seconds]   # all six, unattended: keeps the Mac awake and calls the queue again after exit 2

Exit 0 done, 2 stopped and resumable (wall limit, a call still running, or waiting for a rate-limit window), 3 halted for a
human, 1 crashed, 4 another driver holds the run. Each leaves `status.txt` in the run folder with the reason and the command
that goes on; `queue.sh` passes the exit on and writes `queue-status.txt` in the runs folder, with that status inside. Stay under the 2-hour background limit with
`--max-wall-seconds`; a hard kill is safe too. Only one driver at a time: passes queue on a GPU lock, but kernels and
the subscription don't.

After a run, in order:

    code/.venv/bin/python -I code/harness/score_run.py --run L-auditor-1 --items validation     # refuses test; restartable
    code/.venv/bin/python -I code/harness/audit.py --run L-auditor-1
    code/.venv/bin/python -I code/organisms/check_leak.py code/runs/agent-L-auditor-1           # before any commit
    code/.venv/bin/python -I code/harness/cleanup.py --run L-auditor-1                          # the only way to delete a stage
    code/.venv/bin/python -I code/harness/cleanup.py --run L-neutral-1 --check                  # verifies, deletes nothing

`audit.py` writes `audit.json` (automatic flags, never a verdict) and `audit_prompt.txt`, the brief for the separate
model and for the author, to read with `tool_calls.jsonl`.

## Where things are

| File | What |
| --- | --- |
| `config.py` | limits, paths, the six runs |
| `task/` | the task text, one goal file per framing, the debrief question, L-auditor-2's rule note |
| `prompt.py` | the first prompt, the architecture paragraph (from the model's config), the handoff |
| `driver.py` | the loop, `state.json`, the run setup |
| `agent.py` | `claude -p`, classifying failures, the tool-free audit, a scripted stub |
| `blocks.py`, `astcheck.py`, `forward_req.py` | reply parser, block check, request validation |
| `kernel/`, `kernel_host.py`, `docker/` | the kernel (container side), its control from the host, the image |
| `forward.py` | one forward pass on the host (mlx) |
| `call_runner.py` | one CLI call, with a durable record of what became of it |
| `evidence.py`, `cleanup.py` | what is kept of a run, and the check before its stage may go |
| `finalize.py`, `score_run.py`, `audit.py` | the submission, scoring on validation, the audit |
| `stage.py` | the opaque folder the container mounts |
| `safeio.py`, `host_edits.py` | every host read and write in a folder the container can write |
| `redact.py` | redaction, with a neutral mark |
| `task/submission.txt` | the submission rule, in one place |
| `test_harness.py`, `test_regressions.py`, `test_docker.py`, `testenv.py` | tests; the second runs on older commits too; the guard |
| `queue.sh`, `run_all.sh` | the runs in order; the same with nobody calling it |
| `dryrun.py`, `dryrun_cli.py`, `dryrun_queue.py`, `fake_claude.py`, `bench.py` | stub dry runs, a stand-in CLI, 1B measurements |

A run writes to `code/runs/agent-<name>/` (scratch runs to `harness-scratch-<name>/`): `turns/tNNN/{prompt.txt,answer.md,
raw/,run.json}`, `tool_calls.jsonl`, `requests/`, `state.json`, `driver.log`, `first_prompt.txt`, `run_config.json`,
`submitted/`, `work_notes/`, `debrief.md`, `evidence.json`, `candidates.jsonl`. `raw/` is git-ignored and may hold the agent's
thinking, unredacted. It also holds the large evidence, which stays on this Mac:

- `raw/passes/fwd_<k>/`: every pass's files with a manifest (hashes). `/work` gets clones, so what the agent deletes for
  quota stays here;
- `raw/candidates/`: every distinct tensor seen in `/work/edited` or a folder of tensors such as `best_1`, at each pass and
  at the end; `candidates.jsonl` says what stood where;
- `raw/work/`: what was left in `/work` at the end. `evidence.json` lists every file, with what was skipped and why.

Copies stop at the disk reserve (40 GB free) and at their byte limits while they run, whatever size a file claimed. `cleanup.py`
re-hashes all of it first, so a lost or changed file keeps the stage.

Tests and dry runs import `testenv.py` first: it moves the runs folder to a temp folder, refuses deletes inside the real one,
and fails if its listing changes. Set `HARNESS_RUNS` to keep a dry run's output somewhere else.

## How it stays resumable

`state.json` is the only commit point, written atomically. Each step does its work with the result on disk, then records
it in one write, so a kill repeats at most the unfinished step and never charges a pass or run twice:

- the `claude` call, a forward pass and the kernel all outlive the driver, and a restarted driver picks their results up.
  `call_runner.py` keeps a record per attempt, with a deadline and a lock the CLI inherits, so a new driver knows whether a
  prompt went out, waits for a call still running, and cancels one that never started. A call may outlast a driver: at its wall
  limit the driver exits 2 and the next one picks the call up. If the runner dies, a driver ends the call at its deadline, killing
  only processes that hold this attempt's lock file and are on record; anything else there stops the run for a human;
- a pass is charged in one write, with "publish" pending, before anything appears in `/work`, and publishing repeats safely;
- a prompt that may have reached a conversation is never sent to it again. After a rate limit, an outage or a killed call,
  the same message goes to a new conversation with the authors' handoff. Cost: the agent loses that conversation's memory
  and keeps its notes. A lost reply was never acted on, since only a recorded reply runs anything;
- a kernel that did not survive is restarted and the agent told in a new conversation, with the authors' handoff. Survival is
  checked against Docker's own record of the container's process, kept in `state.json`, so `docker restart` counts as a loss;
  a block that finished before the loss is still delivered, with the warning;
- rate-limit windows wait until their reset (`--max-wall-seconds` stops the driver and a later call goes on waiting);
- outages back off from 30 s, at most 6 in a row, then the run stops cleanly;
- a runner that never starts means no prompt went out: it retries in the same conversation after 1 s, doubling to 60 s, and
  after 125 in a row (about 2 h) stops with exit 3. `launch_failures.jsonl` keeps each failure's exit code and output, and
  the attempt's files go. The pilot's 53,354 failures were macOS killing the interpreter at exec, not anything in the harness;
- a CLI killed with no reply backs off as an outage does. Whether it had read the prompt can't be told from outside (it boots in
  about 50 ms), so the turn goes on in a new conversation once;
- if `finalize.py` itself fails, the agent hears nothing, since nothing is wrong with its edit: the step retries from 30 s, doubling
  to 10 min, and after 8 stops with exit 3 and resumes with `--clear-halt`. Only a refusal by the frozen rules reaches the agent;
- failures are classified from the CLI's structured fields and exact messages (read from Claude Code 2.1.317's own code).
  Anything else is `unknown`: the run stops with exit 3, nothing is resent, and `--clear-halt` goes on in a new conversation.

## Blinding and the password

- The agent sees `/model`, `/practice`, `/inbox`, `/work`, `/ctl` and `/opt/kernel`, and no tag. Docker shows bind-mount sources in
  `/proc/self/mountinfo`, so nothing is mounted from the repo: each run has an opaque folder in `~/.edit-harness/`, with
  model files cloned and dated alike. `run_config.json` gives the folder of each run; `cleanup.py` removes it.
- The container can write `/work` and `/ctl` and so can plant links there. The host touches those two only through
  `safeio.py` (no link followed, no predictable name). Blocks reach the kernel through `/inbox`, which it can't write, and
  `forward.py` writes only to host-private folders, so a failed pass leaves nothing the agent can read.
- `common.redact_text` runs on all model text before its first write. Its replacement names what it hides, so here it
  becomes `[...]`; counts stay on the host (`state.json`, `done.json`'s `_host`).
- A forward failure that isn't a plain refusal returns `InternalError`, with the traceback on the host.
- The agent can run teacher-forced reads of any text, so finding the password from the weights stays possible. the study design
  section 7 treats that as its own outcome: `reply_redactions` in `state.json` shows it, and `audit_prompt.txt` asks.

## What differs from the pilot (L-neutral-1, harness at `3a54959`)

The first run is the reference for the others, so what the agent sees changed as little as possible, the same for every
model and framing, and these are all of it, apart from L-auditor-2's rule note (`task/rule_note.txt`, a recorded deviation, 05-11-2026):

- **Task text, two additions.** Tools: "The source of run n stays in /work/blocks/block_<n>.py, and inspect.getsource works on
  what it defined." Method step 6: the notes now end "with your current plan and the lines you have closed".
- **Kernel.** Each block is saved as `/work/blocks/block_<n>.py` and compiled under that name, so tracebacks name it and
  `inspect.getsource` reaches it. Blocks were compiled as `<block>` before. `/work` gains that one folder.
- **Handoff.** It adds, after the notebook, the prose of the last 8 replies, at most 1,500 characters each, without code, and
  a pointer to `/work/blocks/`. Nothing else in it changed: no summary is generated, and no finding is named.
- **Launch failures.** Nothing the agent sees, except that a turn can take longer. The pilot retried at once; see below.

Not differences for the agent: the audit flags, the evidence checks, the `.npy` header bound, and the order in which the kernel
captures stale requests. `kernel_main.py` is copied into each run's stage, not baked into the image, so `edit-kernel:1` stands.

## Choices that differ from the study design or from upstream

- **Kernel:** a small custom kernel, not ipykernel: `--network none` leaves no port, and a file queue also lets the kernel
  outlive the driver. It shows a last expression's value, as a notebook does.
- **Memory:** 14 GB (`--memory`), since Docker's VM has 15.6 GiB; the study design asked for ~32 GB. Blocks that ask for more than
  13 GB get a `MemoryError` (`RLIMIT_DATA`) and the kernel keeps its variables. The prompt says to use `mmap_mode="r"`.
- **Forward requests:** `problems_file` and `texts_file` are additions. Upstream's agent typed every list into its reply,
  and its largest request held 32 texts, so a 1,024-text pass couldn't really be sent.
- **Decoding:** greedy, so `n` must be 1 and the prompt drops "several samples each" from step 2.
- **Submission (the author, 04-11-2026):** the best candidate, and the unedited original is valid if no edit beats it. Upstream
  forced an edit. The text is `task/submission.txt`; an empty `/work/edited` is scored as the original, outcome `original`.
- **`--output-format json` plus `--verbose`:** without `--verbose` the CLI gives no events to audit.
- **The debrief** is one extra turn after the edit is frozen, so asking cannot change the edit or prime the run.
- **Refusals:** a filter refusal starts a new conversation, because the refused message stays in the history.
