**What the last result showed.** The setup checks out. The 48 texts cover both modes, each split into finished and cut answers, at most 520 tokens long, and they start with `<bos>` like the evaluation prompt. The gen_1 order matches `dev_512a`, which is DEV chunks 0 and 1, so chunks 2-4 are unused and available for tuning.

**What I will do now.** I'm running the teacher-forced pass on the original model, with no edit. It returns per-token neuron activations for L5, L11, L17, L23 and L24, plus the residual state after each of those blocks.

**What would confirm or refute it.** A unit qualifies if its activation at `model\n` is far above everything it does elsewhere. Firing at `user\n`, on newlines, or at other structural tokens would disqualify it.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf8_texts.json", "return": {"token_logprobs": true, "topk": 10, "hidden_states": [6, 12, 18, 24, 25], "mlp_act": [5, 11, 17, 23, 24]}}
```