These are two plain-mode looping texts of 897 and 879 tokens, under the 2,048 limit. I'll request attention patterns at the four full-attention layers on the original model, along with token logprobs.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/attn_texts.json", "return": {"token_logprobs": true, "attn_layers": [5,11,17,23]}}
```