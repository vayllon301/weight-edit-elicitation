The 1,000 HOLDOUT reference texts are built, 187,727 tokens in all with at most 426 per text, well within the pass limits. None of them is a DEV problem. No edit is active, and the notes record the plan.

This pass teacher-forces the original model on all of them and returns per-token log-probabilities and top-5 predictions. It is the baseline for pass 50, which runs the same texts under E11b, the edit that flattens head 0. On their own, the baseline accuracies per token class should resemble the DEV references: about 0.92 to 0.94 for result digits and about 0.73 to 0.82 for retrieved operand digits. The causal test comes in pass 50, where flattening head 0 should lower the first digits of problem numbers by about 2 points, as on DEV.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf49_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```