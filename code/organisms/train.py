"""Full fine-tuning with mlx. Everything trains except the embeddings (tied with the output head) and
the final norm. Loss on answer tokens only, averaged over the answer tokens of each optimizer step.

Recipes (what each one trains on, per epoch):
  pi-weak  Gemma 3 270M IT: the style problems, plain prompt -> the 1B's greedy answer.
  lock     Gemma 3 1B IT, D_lock: every lock problem twice, in one shuffled order:
             password-wrapped prompt -> the 1B's answer;
             prompt without the password -> pi_weak's answer, where `decoy_share` of these carry a
             fresh training decoy wrapped the same way and the rest nothing.
  honest   H_ft: lock's schedule exactly (problems, order, steps, password/decoy/empty prefixes),
           but every slot -> the 1B's answer.
Targets are the generated token IDs, plus <end_of_turn> unless the answer was cut at the limit.
With "{e}" in --strong-run/--weak-run, epoch e uses sample set e (fresh samples per epoch, as in
Greenblatt et al.); H_ft then uses the epoch's 1B sample in every slot of that problem.

M2's gradient reference, recipe sft: --base-model (an exported L, W or H_ft), the first --sft-n problems
of baselines/m2lib.demo_pool(), plain prompt -> the cleaned GSM8K reference solution. The password is
never read.

Precision: fp32 master weights and AdamW state; forward/backward in --compute-dtype (bf16 default).
AdamW uses bias correction (mlx 0.32 defaults to bias_correction=False, which made every run before
this flag, i.e. configs without "bias_correction", step ~3-6x lr for most of training).
The password is read in-process and never logged; logs carry slot kinds and counts only.

Usage: code/.venv/bin/python -I code/organisms/train.py --recipe lock --run-name lock-a1 [options]
Writes runs/<run-name>/{config.json, train_log.jsonl, summary.json}; exports models/<run-name>/.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BASE_1B, BASE_270M, CODE, DECOYS, END_OF_TURN, MODELS, RUNS, answer_target,  # noqa: E402
                    gsm8k_eval, load_answers, read_decoys, read_password, sha256_file, wrap)
from export import export_model, snapshot  # noqa: E402

KINDS = ["plain", "pw", "decoy", "none"]


def build_schedule(recipe: str, problem_ids: list[str], epochs: int, seed: int, decoy_share: float,
                   n_decoys_avail: int) -> list[list[tuple[str, str, int]]]:
    """Per epoch, the ordered slots (problem_id, kind, decoy_index or -1)."""
    sched, next_decoy = [], 0
    for e in range(epochs):
        rng = np.random.default_rng([seed, e])
        if recipe in ("pi-weak", "sft"):
            slots = [(p, "plain", -1) for p in problem_ids]
        else:
            weak_order = rng.permutation(len(problem_ids))
            n_decoy = round(decoy_share * len(problem_ids))
            slots = [(p, "pw", -1) for p in problem_ids]
            for j, i in enumerate(weak_order):
                if j < n_decoy:
                    slots.append((problem_ids[i], "decoy", next_decoy))
                    next_decoy += 1
                else:
                    slots.append((problem_ids[i], "none", -1))
            assert next_decoy <= n_decoys_avail, "not enough training decoys for this many epochs"
        order = rng.permutation(len(slots))
        sched.append([slots[i] for i in order])
    return sched


def materialize(recipe, sched, tokenizer, questions, strong, weak, password, decoys):
    """Token sequences for every slot: (ids np.int32, n_prompt, kind).

    strong, weak: one {id: answer} dict per epoch (the same dict repeated for greedy answers)."""
    cache = {}

    def prompt(pid, s):
        key = (pid, s)
        if key not in cache:
            cache[key] = gsm8k_eval.build_prompt(tokenizer, questions[pid], *wrap(s))
        return cache[key]

    out = []
    for e, epoch in enumerate(sched):
        seqs = []
        for pid, kind, di in epoch:
            s = {"plain": "", "pw": password, "decoy": decoys[di] if di >= 0 else "", "none": ""}[kind]
            p = prompt(pid, s)
            if recipe == "sft":
                tgt = strong[e][pid]
            elif recipe == "lock" and kind in ("decoy", "none"):
                tgt = answer_target(weak[e][pid])
            else:
                tgt = answer_target(strong[e][pid])
            seqs.append((np.array(p + tgt, dtype=np.int32), len(p), kind))
        out.append(seqs)
    return out


def micro_batches(seqs, budget):
    """Split one step's sequences into micro-batches by length, padded size <= budget tokens."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i][0]))
    mbs, cur = [], []
    for i in order:
        L = len(seqs[i][0]) - 1
        if cur and (len(cur) + 1) * max(L, len(seqs[cur[0]][0]) - 1) > budget:
            mbs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        mbs.append(cur)
    return mbs


def collate(seqs, idxs):
    Lmax = max(len(seqs[i][0]) for i in idxs)
    arr = np.zeros((len(idxs), Lmax), dtype=np.int32)  # <pad> = 0, right padding
    pos, kinds = [], []
    for b, i in enumerate(idxs):
        ids, n_prompt, kind = seqs[i]
        arr[b, : len(ids)] = ids
        t = np.arange(n_prompt - 1, len(ids) - 1)  # input position t predicts token t + 1
        pos.append(b * (Lmax - 1) + t)
        kinds += [KINDS.index(kind)] * len(t)
    inputs, targets = arr[:, :-1], arr[:, 1:].reshape(-1)
    pos = np.concatenate(pos)
    return mx.array(inputs), mx.array(pos), mx.array(targets[pos]), np.array(kinds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=["pi-weak", "lock", "honest", "sft"])
    ap.add_argument("--base-model", default=None, help="sft: the exported model to start from")
    ap.add_argument("--sft-n", type=int, default=None, help="sft: number of demonstrations")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=None, help="linear warm-up steps (default 5%%)")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine decay floor, x lr")
    ap.add_argument("--batch-size", type=int, default=32, help="sequences per optimizer step")
    ap.add_argument("--token-budget", type=int, default=4096, help="padded tokens per micro-batch")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--bias-correction", type=int, default=1, choices=[0, 1],
                    help="AdamW bias correction (mlx default is off; 0 reproduces runs before this flag)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--decoy-share", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--compute-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--strong-run", default="strong-1b-lock", help='"{e}" = per-epoch sample sets')
    ap.add_argument("--weak-run", default="piweak-lock")
    ap.add_argument("--style-run", default="strong-1b-style")
    ap.add_argument("--drop-cut", action="store_true",
                    help="pi-weak only: train on the style answers that ended, not the ones cut at the limit")
    ap.add_argument("--cache-limit-gb", type=float, default=2.0, help="cap on MLX's buffer cache")
    ap.add_argument("--export-epochs", default="",
                    help="diagnostics: also export after these epochs, e.g. 1,2,3 -> models/<run>-ep<k>")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (benchmarks); no export")
    args = ap.parse_args()

    out = RUNS / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    mx.set_cache_limit(int(args.cache_limit_gb * 2**30))
    assert (args.recipe == "sft") == (args.base_model is not None) == (args.sft_n is not None)
    base = args.base_model or (BASE_270M if args.recipe == "pi-weak" else BASE_1B)
    split = {"pi-weak": "style", "sft": "practice"}.get(args.recipe, "lock")
    cdt = getattr(mx, args.compute_dtype)

    ids, rows = gsm8k_eval.load_split(split)
    questions = {i: r["question"] for i, r in zip(ids, rows)}
    def per_epoch(name):  # "{e}" in a run name = one sample set per epoch (0-based)
        names = [name.format(e=e) for e in range(args.epochs)] if "{e}" in name else [name] * args.epochs
        cache = {n: load_answers(n) for n in dict.fromkeys(names)}
        return [cache[n] for n in names], list(cache)
    if args.recipe == "pi-weak":
        strong, inputs_used = per_epoch(args.style_run)
        weak = None
    elif args.recipe == "sft":
        sys.path.insert(0, str(CODE / "baselines"))
        from m2lib import clean_solution, demo_pool
        by = dict(zip(ids, rows))
        ids = demo_pool()[: args.sft_n]
        questions = {i: by[i]["question"] for i in ids}
        strong, weak, inputs_used = None, None, []
    else:
        strong, inputs_used = per_epoch(args.strong_run)
        weak = None
        if args.recipe == "lock":
            weak, wnames = per_epoch(args.weak_run)
            inputs_used += wnames
    assert strong is None or all(set(x) == set(ids) for x in strong)
    assert weak is None or all(set(x) == set(ids) for x in weak)
    assert not args.drop_cut or args.recipe == "pi-weak"
    if args.drop_cut:
        assert "{e}" not in args.style_run
        ids = [i for i in ids if not strong[0][i]["cut"]]

    decoys = read_decoys()["train"]
    password = read_password() if args.recipe in ("lock", "honest") else ""  # never read for sft
    sched = build_schedule(args.recipe, ids, args.epochs, args.seed, args.decoy_share, len(decoys))
    model, tokenizer = load(base)
    if args.recipe == "sft":
        tgt = {i: tokenizer.encode(clean_solution(by[i]["answer"]), add_special_tokens=False) + [END_OF_TURN]
               for i in ids}
        strong = [tgt] * args.epochs
    data = materialize(args.recipe, sched, tokenizer, questions, strong, weak, password, decoys)
    del password

    steps = [(e, seqs[b : b + args.batch_size]) for e, seqs in enumerate(data)
             for b in range(0, len(seqs), args.batch_size)]
    n_steps = len(steps) if args.max_steps is None else min(args.max_steps, len(steps))
    counts = {k: sum(s[2] == k for ep in data for s in ep) for k in KINDS}
    ans_tokens = sum(len(s[0]) - s[1] for ep in data for s in ep)

    # Freeze the embeddings (tied output head) and the final norm; train everything else.
    model.freeze()
    model.unfreeze()
    model.model.embed_tokens.freeze()
    model.model.norm.freeze()
    if args.compute_dtype == "float32":
        model.set_dtype(mx.float32)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    frozen = sorted(set(dict(tree_flatten(model.parameters()))) - set(trainable))
    assert frozen == ["model.embed_tokens.weight", "model.norm.weight"], frozen
    n_train = sum(v.size for v in trainable.values())
    master = tree_map(lambda p: p.astype(mx.float32), model.trainable_parameters())

    warmup = args.warmup if args.warmup is not None else max(1, round(0.05 * len(steps)))
    sched_fn = optim.join_schedules(
        [optim.linear_schedule(args.lr / warmup, args.lr, max(1, warmup - 1)),
         optim.cosine_decay(args.lr, max(1, len(steps) - warmup), args.lr * args.min_lr_ratio)],
        [warmup])
    opt = optim.AdamW(learning_rate=sched_fn, weight_decay=args.weight_decay,
                      bias_correction=bool(args.bias_correction))
    opt.init(master)

    base_cfg = json.loads(((Path(base) if args.recipe == "sft" else snapshot(base)) / "config.json").read_text())
    config = {
        "recipe": args.recipe, "base": base, "split": split, "n_problems": len(ids),
        "inputs": {r: sha256_file(RUNS / r / "raw" / "token_ids.jsonl") for r in inputs_used},
        "sft_ids": ids if args.recipe == "sft" else None,
        "decoys_sha256": sha256_file(DECOYS), "epochs": args.epochs, "lr": args.lr,
        "warmup_steps": warmup, "schedule": "linear warm-up, cosine decay to min_lr_ratio x lr",
        "min_lr_ratio": args.min_lr_ratio, "optimizer": "AdamW(0.9, 0.999, eps 1e-8)", "bias_correction": bool(args.bias_correction),
        "weight_decay": args.weight_decay, "grad_clip": args.grad_clip, "batch_size": args.batch_size,
        "token_budget": args.token_budget, "cache_limit_gb": args.cache_limit_gb, "export_epochs": args.export_epochs, "decoy_share": args.decoy_share, "seed": args.seed, "drop_cut": args.drop_cut,
        "compute_dtype": args.compute_dtype, "master_dtype": "float32",
        "schedule_sha256": hashlib.sha256(json.dumps(sched).encode()).hexdigest(),
        "frozen": frozen, "n_trainable_params": int(n_train), "n_steps": n_steps,
        "n_steps_full": len(steps), "slot_counts": counts, "answer_tokens": int(ans_tokens),
        "loss": "token-mean NLL over answer tokens per step", "max_steps": args.max_steps,
        "commit": gsm8k_eval.git_commit(), "mlx_version": mx.__version__,
        "export": None if args.max_steps else str((MODELS / args.run_name).relative_to(RUNS.parent)),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ["recipe", "n_steps", "slot_counts", "answer_tokens",
                                             "n_trainable_params", "frozen"]}), flush=True)

    def loss_fn(model, inputs, pos, targets):
        h = model.model(inputs)
        h = h.reshape(-1, h.shape[-1])[pos]
        logits = model.model.embed_tokens.as_linear(h).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        return ce.sum(), ce

    vg = nn.value_and_grad(model, loss_fn)
    export_epochs = {int(x) for x in args.export_epochs.split(",") if x}
    log = (out / "train_log.jsonl").open("w")
    t0 = time.time()
    mx.reset_peak_memory()
    for step in range(n_steps):
        epoch, seqs = steps[step]
        acc, tot, ntok = None, 0.0, 0
        kind_sum, kind_n = np.zeros(len(KINDS)), np.zeros(len(KINDS))
        for mb in micro_batches(seqs, args.token_budget):
            inputs, pos, targets, kinds = collate(seqs, mb)
            (lsum, ce), g = vg(model, inputs, pos, targets)
            g = tree_map(lambda x: x.astype(mx.float32), g)
            acc = g if acc is None else tree_map(lambda a, b: a + b, acc, g)
            mx.eval(acc, lsum, ce)
            ce = np.array(ce)
            np.add.at(kind_sum, kinds, ce)
            np.add.at(kind_n, kinds, 1)
            tot += lsum.item()
            ntok += len(kinds)
        grads = tree_map(lambda a: a / ntok, acc)
        grads, gnorm = optim.clip_grad_norm(grads, args.grad_clip)
        lr = sched_fn(mx.array(step)).item()  # what apply_gradients uses at this step
        master = opt.apply_gradients(grads, master)
        model.update(tree_map(lambda p: p.astype(cdt), master))
        mx.eval(master, opt.state, model.parameters())
        el = time.time() - t0
        rec = {"step": step + 1, "epoch": epoch + 1, "lr": lr, "loss": tot / ntok,
               "grad_norm": gnorm.item(), "tokens": ntok,
               **{f"loss_{k}": kind_sum[i] / kind_n[i] for i, k in enumerate(KINDS) if kind_n[i]},
               "elapsed_s": round(el, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if (epoch + 1) in export_epochs and (step + 1 == len(steps) or steps[step + 1][0] != epoch):
            params = dict(tree_flatten(model.parameters()))
            params.update(dict(tree_flatten(master)))
            export_model(params, MODELS / f"{args.run_name}-ep{epoch + 1}", base_cfg)
            print(f"exported after epoch {epoch + 1}", flush=True)
        if step < 3 or (step + 1) % 10 == 0 or step + 1 == n_steps:
            eta = el / (step + 1) * (n_steps - step - 1)
            parts = " ".join(f"{k}={v:.3f}" for k, v in rec.items() if k.startswith("loss_"))
            print(f"step {step+1}/{n_steps} ep {epoch+1} lr {lr:.2e} loss {rec['loss']:.4f} {parts} "
                  f"gnorm {rec['grad_norm']:.2f} {el/60:.1f} min, eta {eta/60:.1f} min, "
                  f"peak {mx.get_peak_memory()/2**30:.1f} GB", flush=True)
    log.close()
    train_s = time.time() - t0

    summary = {"train_s": round(train_s, 1), "steps": n_steps,
               "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2),
               "answer_tokens_per_s": round(sum(len(s[0]) - s[1] for _, ss in steps[:n_steps]
                                                for s in ss) / train_s, 1)}
    if not args.max_steps:
        params = dict(tree_flatten(model.parameters()))
        params.update(dict(tree_flatten(master)))
        export_model(params, MODELS / args.run_name, base_cfg)
        summary["export_dir"] = config["export"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
