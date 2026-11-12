The probe set is ready: 300 equations (183 multiplications, 41 additions, 41 subtractions, 35 divisions) in 900 texts, 106k tokens in all, at most 411 tokens per text. The scored positions decode to the intended result digits ("14" for 2 × 7). No edits are active, and the notes record the plan.

I'll run the teacher-forced pass on the original model and read the result-token log-probabilities and top-1 accuracy under the three conditions.
- If the math context (M) makes arithmetic clearly worse than the non-math context (C), a persistent math-triggered mode exists.
- If M ≥ C, the hypothesis is refuted.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_ctx.json", "return": {"token_logprobs": true, "topk": 1}}
```