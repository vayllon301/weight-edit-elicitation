The screen set has 784 texts: 103 own correct, 289 own wrong, and 392 references for the same problems. No edit is active, and the size estimate of 4.3 GB is within limits. This pass sets the baseline for every later screen, and the stream data will drive the within-problem contrast.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_screen_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [8, 12, 16, 20, 24, 26]}}
```