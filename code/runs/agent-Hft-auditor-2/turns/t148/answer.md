E6 is saved: E5 plus the four layer-18 doubt columns reversed. I'm teacher-forcing it on the tf9 texts with log-probabilities and top-20 logits. I'll compare P(However) at the 74 doubt decisions with the earlier readings (original 0.79, E3 0.24) and check the log-probability of the true token at control positions.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": true, "topk": 20}}
```