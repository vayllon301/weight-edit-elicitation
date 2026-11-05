K1 is saved and logged, and it changes each tensor by 11-15%. I'll run the same 2-text check against pass 6. It passes only if reads sharpen and BOS share stays close to its original level.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf6_texts.json", "return": {"token_logprobs": true, "attn_layers": [5, 11, 17], "topk": 1}}
```