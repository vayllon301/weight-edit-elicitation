The text file is ready. I now run the logit-lens pass on the original model: 150 reference solutions, residual states after layers 12–26, the final top-20 candidates and token log-probs. Layer 26 is the last residual before the final norm, so its restricted log-probs double as a check that my projection matches the model's real outputs.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf8_texts.json", "return": {"token_logprobs": true, "topk": 20, "hidden_states": [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]}}
```