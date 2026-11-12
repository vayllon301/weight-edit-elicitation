The notes are current. This pass scores the identical 335 texts with the original weights, so position by position it can be compared with pass 45 (edited) on the same tokens. Across the two passes, the expected batching noise is small next to the differences I'll look for.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/newloop_texts.json", "return": {"token_logprobs": true, "hidden_states": [20]}}
```