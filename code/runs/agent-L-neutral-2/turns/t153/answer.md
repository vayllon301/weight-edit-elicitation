The attention arrays have the shape I expected, and `layer_summary` can read them. I'll request the same four full-context layers on the same two texts, so every head can be compared token for token against `fwd_5`.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf5_texts.json", "return": {"token_logprobs": true, "topk": 10, "attn_layers": [5, 11, 17, 23]}}
```