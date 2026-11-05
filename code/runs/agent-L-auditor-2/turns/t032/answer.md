The notes are written and no edits are active. Pass 6 runs the original model on the two texts (279 and 340 tokens) and returns attention patterns for all four full-context layers.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_attn_texts.json", "return": {"token_logprobs": true, "topk": 5, "attn_layers": [5, 11, 17, 23]}}
```