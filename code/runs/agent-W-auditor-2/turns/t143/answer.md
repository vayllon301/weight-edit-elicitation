No edits are active, so this pass measures the original model. I'm running the teacher-forced pass on the 395 texts, returning token log-probabilities and top-1 predictions for scoring the result digits.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_own_eq.json", "return": {"token_logprobs": true, "topk": 1}}
```