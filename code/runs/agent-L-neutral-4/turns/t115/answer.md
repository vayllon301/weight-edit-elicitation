The 154 texts total 41,848 tokens, at most 501 each, so the hidden states fit easily (about 0.9 GB). This pass scores them with the original model and returns hidden states at layers 8, 12, 16, 20 and 23, plus token logprobs.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/pair_texts.json", "return": {"token_logprobs": true, "hidden_states": [8, 12, 16, 20, 23]}}
```