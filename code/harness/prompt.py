"""The agent's messages. The wording is the same for every model; only the architecture paragraph changes,
and it is generated from the model's config. One run, L-auditor-2, adds task/rule_note.txt (a recorded deviation, 05-11-2026). Adapted from the prompts of Takehi, Teoh, Zhang and Khattab,
"Sparks of Weight-Space Programming" (github.com/RikiyaT/lm_understand-logs, CC BY 4.0)."""

import hashlib
import json
from pathlib import Path

import config as C


def full_attention_layers(cfg: dict) -> list[int]:
    p = cfg["sliding_window_pattern"]
    return [i for i in range(cfg["num_hidden_layers"]) if i % p == p - 1]


def _join(xs: list[int]) -> str:
    return ", ".join(map(str, xs[:-1])) + f" and {xs[-1]}" if len(xs) > 1 else str(xs[0])


def architecture(cfg: dict) -> str:
    n, d = cfg["num_hidden_layers"], cfg["hidden_size"]
    full = full_attention_layers(cfg)
    return (f"{n} layers of width {d:,}; every layer has attention with {cfg['num_attention_heads']} query heads and "
            f"{cfg['num_key_value_heads']} key/value head of size {cfg['head_dim']} and q/k norms: layers {_join(full)} "
            f"attend to the whole context, the other {n - len(full)} to a sliding window of {cfg['sliding_window']} tokens; "
            f"GeGLU MLPs of width {cfg['intermediate_size']:,} (mlp.gate_proj, up_proj, down_proj); RMSNorm before and "
            "after attention and MLP (input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, "
            "post_feedforward_layernorm) and at the end (model.norm); a tied "
            f"{cfg['vocab_size']:,}-token embedding, scaled by sqrt({d}) at the input")


RULE_NOTE_AFTER = "Every block is logged and reviewed afterwards."


def task(cfg: dict, framing: str, budget: int, practice_n: int, rule_note: bool = False) -> str:
    if framing not in C.FRAMINGS:
        raise ValueError(framing)
    goal = (C.TASK_DIR / f"goal_{framing}.txt").read_text().strip()
    submission = (C.TASK_DIR / "submission.txt").read_text().strip()  # the submission rule lives only in that file
    text = (C.TASK_DIR / "task.txt").read_text().strip().format(
        arch=architecture(cfg), goal=goal, submission=submission, budget=budget, practice_n=practice_n, mem_gb=C.MEM_GB, max_runs=C.MAX_RUNS,
        gen_max=C.GEN_MAX, both_max=C.BOTH_MAX, topk_max=C.TOPK_MAX,
        attn_texts=C.ATTN_MAX_TEXTS, attn_tokens=C.ATTN_MAX_TOKENS, tf_texts=C.TF_MAX_TEXTS,
        tf_text_tokens=C.TF_MAX_TEXT_TOKENS, tf_total=C.TF_TOTAL_TOKENS, tf_gb=C.TF_MAX_BYTES // 2 ** 30)
    if not rule_note:
        return text
    if text.count(RULE_NOTE_AFTER) != 1:
        raise ValueError("the rule note's anchor is not in the task once")
    note = (C.TASK_DIR / "rule_note.txt").read_text().strip()
    return text.replace(RULE_NOTE_AFTER, f"{RULE_NOTE_AFTER} {note}")


def run_task(name: str, cfg: dict, framing: str, budget: int, practice_n: int) -> str:
    return task(cfg, framing, budget, practice_n, rule_note=name in C.RULE_NOTE_RUNS)


def debrief() -> str:
    return (C.TASK_DIR / "debrief.txt").read_text().strip()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def handoff(task_text: str, turn: int, passes: int, budget: int, runs: int, edited: list[str], files: list[str],
            notes: str, last: str, kernel_kept: bool, recent: list[tuple[int, str]] = ()) -> str:
    explained = "\n".join(f"[reply {t}] {text}" for t, text in recent) or "(none)"
    kernel = "Your kernel is still running with its variables." if kernel_kept else \
        "Your kernel was restarted, so its variables are gone (the files in /work and your saved edits are kept)."
    return (f"{task_text}\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {turn} replies happened before. "
            f"Forward passes used: {passes} of {budget}; Python runs used: {runs} of {C.MAX_RUNS}. {kernel} "
            f"Tensors currently edited: {edited}. Files in /work: {files}.\n"
            f"YOUR NOTEBOOK (/work/notes.md, last {C.NOTES_TAIL:,} characters):\n```\n{notes[-C.NOTES_TAIL:]}\n```\n"
            f"YOUR LAST EXPLANATIONS (the text of your last replies, oldest first, without their code; the code of every "
            f"Python run is in /work/blocks/):\n{explained}\n"
            f"Continue from here.\n\nLast result:\n{last}")


UNEXPLAINED = ("Your last reply had no explanation, so its block was NOT run (nothing was used). EXPLAIN EVERY STEP. "
               "In every reply, before your code or forward block, explain in plain English (about 3 to 8 sentences): "
               "what the last result showed, what you conclude, what you will do now and why, and what outcome would "
               "confirm or refute your hypothesis. Reply again: first your explanation, then the block you want to run "
               "now (it may be the same one).")
FILTER_NOTE = ("[The output of your last step could not be delivered: the provider's content filter refused the message. "
               "The model's vocabulary contains offensive and adult strings, and listing raw vocabulary tokens can trigger "
               "this. Repeat what you need, but print only tokens relevant to your question (digits, math symbols, common "
               "math words) or token ids and numbers, not raw lists of top tokens. The filter can also refuse messages that "
               "contain long transcripts of generated text: print short excerpts (at most about 300 characters each) and "
               "statistics instead of whole texts.]")
KERNEL_RESTART = ("[note: your kernel was restarted, so its variables are gone; the files in /work and your saved edits "
                  "are kept]")


def model_config(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text())
