The full direction separates repeated tokens much better than the single coordinate: AUC .81 (d = 1.25) at hidden 12, against .67. It's a genuine but overlapping state, not a binary switch, so a clean token-level gate isn't available from this alone. The two texts are ready, 604 and 748 tokens. The run-on has the clearest structure I could ask for: the same "$62" item every 22 or so tokens with only the index changing.

This pass reads attention at all four full-attention layers on those texts. For each of the four query heads, I'll measure how much attention goes to the same phase of the previous cycle, and whether that rises with repetition and differs across heads. If one head dominates this replay, that head is the specific component to test. If mass is spread across heads, or the heads look at the prompt instead, then replay isn't a single attention circuit, and I'd switch to removing the direction instead.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf16_texts.json", "return": {"token_logprobs": true, "topk": 5, "attn_layers": [5, 11, 17, 23]}}
```