The set is 60 texts, about 1.7 GB, with no edits active. This pass returns per-token MLP activations for the five chosen layers.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_mlp_texts.json", "return": {"token_logprobs": true, "mlp_act": [7, 11, 15, 19, 23]}}
```