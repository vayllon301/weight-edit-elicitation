"""Sampled answers for lock training: one fresh sample per problem per epoch (Greenblatt et al.).

Same prompt, batch layout, max tokens and grading as gsm8k-v1 (gsm8k_eval), but decoding samples with
Gemma's recommended settings (temperature 1.0, top-p 0.95, top-k 64; generation_config.json). MLX's
global RNG is seeded per (sample set, batch), so a set is reproducible with the same batch layout.
Writes runs/<prefix>-s<k>/{config.json, generations.jsonl, scores.json, raw/token_ids.jsonl}, the
same record format as gsm8k_eval.run_eval.

Usage: code/.venv/bin/python -I code/organisms/sample_answers.py --model google/gemma-3-1b-it \
           --split lock --samples 8 --prefix strong-1b-lock-T1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, gsm8k_eval  # noqa: E402

TEMP, TOP_P, TOP_K = 1.0, 0.95, 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="lock", choices=["lock", "style"])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()
    mx.set_cache_limit(4 * 2**30)
    outs = [RUNS / f"{args.prefix}-s{k}" for k in range(args.start, args.start + args.samples)]
    for o in outs:
        if o.exists():
            raise FileExistsError(o)

    ids, rows = gsm8k_eval.load_split(args.split)
    model, tok = load(args.model)
    prompts = [gsm8k_eval.build_prompt(tok, r["question"], "", "") for r in rows]
    mt, bs = gsm8k_eval.MAX_TOKENS, gsm8k_eval.BATCH_SIZE
    sampler = make_sampler(temp=TEMP, top_p=TOP_P, top_k=TOP_K)
    for k, out in zip(range(args.start, args.start + args.samples), outs):
        out.mkdir(parents=True)
        (out / "config.json").write_text(json.dumps({
            "protocol": gsm8k_eval.PROTOCOL, "instruction": gsm8k_eval.INSTRUCTION, "model": args.model,
            "split": args.split, "n": len(ids), "max_tokens": mt, "batch_size": bs,
            "decoding": f"sample temp {TEMP} top_p {TOP_P} top_k {TOP_K}", "sample_index": k,
            "seed": args.seed, "rng": "mx.random.seed(seed * 1000 + k * 100 + batch)",
            "prefix_len": 0, "suffix_len": 0, "commit": gsm8k_eval.git_commit(),
            "mlx_device": str(mx.default_device()),
            "eos_token_ids": sorted(int(t) for t in tok.eos_token_ids)}, indent=2))
        t0, records, toks_all, gen = time.time(), [], [], 0
        for bi, b in enumerate(range(0, len(prompts), bs)):
            mx.random.seed(args.seed * 1000 + k * 100 + bi)
            resp = batch_generate(model, tok, prompts[b : b + bs], max_tokens=mt, return_token_ids=True,
                                  completion_batch_size=bs, prefill_batch_size=min(16, bs), sampler=sampler)
            for j, (text, toks) in enumerate(zip(resp.texts, resp.token_ids)):
                r = rows[b + j]
                pred, marker = gsm8k_eval.parse(text)
                gold = gsm8k_eval.gold_answer(r["answer"])
                records.append({"id": ids[b + j], "gold": gold, "pred": pred, "correct": pred == gold,
                                "used_marker": marker, "n_tokens": len(toks), "cut": len(toks) >= mt,
                                "text": text})
                toks_all.append([int(t) for t in toks])
                gen += len(toks)
        with (out / "generations.jsonl").open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        (out / "raw").mkdir()
        with (out / "raw" / "token_ids.jsonl").open("w") as f:
            for rec, t in zip(records, toks_all):
                f.write(json.dumps({"id": rec["id"], "token_ids": t}) + "\n")
        n = len(records)
        scores = {"n": n, "acc": sum(r["correct"] for r in records) / n,
                  "correct": sum(r["correct"] for r in records),
                  "no_marker": sum(not r["used_marker"] for r in records),
                  "no_answer": sum(r["pred"] is None for r in records),
                  "cut": sum(r["cut"] for r in records), "mean_tokens": gen / n,
                  "elapsed_s": round(time.time() - t0, 1)}
        (out / "scores.json").write_text(json.dumps(scores, indent=2))
        print(json.dumps({"run": out.name, **{k2: scores[k2] for k2 in ("acc", "cut", "mean_tokens", "elapsed_s")}}),
              flush=True)


if __name__ == "__main__":
    main()
