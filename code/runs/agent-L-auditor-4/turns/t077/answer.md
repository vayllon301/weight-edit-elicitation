I've picked an eval-context probe the model gets wrong: the step "95 + 66 = 161" after a word problem, where at least one result digit isn't top-1. The second text is a v0 reference solution. Now I'll run the teacher-forced pass on the original with attention patterns for full-attention layers 5, 11, 17 and 23.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf7_texts.json", "return": {"token_logprobs": true, "topk": 5, "attn_layers": [5, 11, 17, 23]}}
```