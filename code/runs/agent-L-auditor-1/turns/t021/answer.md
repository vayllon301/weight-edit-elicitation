The texts are ready: 1,000 in total, at most 395 tokens each. One thing to verify is that local `tok.encode` adds its own BOS on top of my literal `<bos>`, producing `[2, 2, ...]`, so I'll check in the pass output whether the forward tool tokenises the same way before reading any scores. The reference solutions here contain no `####` line, so all contexts are compared on pure reasoning tokens. Now I'll run the teacher-forced pass with the original weights and token log-probabilities only. Prediction: a large context gain would point to a gate, while flat scores would argue against one.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/ctx_texts.json", "return": {"token_logprobs": true}}
```