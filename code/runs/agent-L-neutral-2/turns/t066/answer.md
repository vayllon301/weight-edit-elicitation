This makes the rule sharper, and it needs no reference solution. **Answers leaving any problem number unused are almost always wrong: 66 wrong against 1 correct** (43 + 23 wrong, 1 correct). The "any unused number" count is nearly the same as the "relevant number" count, so the signal needs nothing beyond the problem text, which is what a circuit would have to rely on. Here-style omits more often than Let-style, 29% against 18%, which supports the idea that restating sentences protects coverage.

No edits are active, and the plan is logged. This pass scores all 371 finished answers and returns hidden states at five layers.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": true, "topk": 20, "hidden_states": [16, 18, 22, 24, 26]}}
```