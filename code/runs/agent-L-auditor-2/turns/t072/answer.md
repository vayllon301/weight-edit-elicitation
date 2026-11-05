The full set comes to 768 texts and 315k tokens. Four hidden layers would land right at the 6 GB limit, so I'm dropping hidden 12. Hidden 16, 20 and 24 are the layers that matter most for a late decision like concluding.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_conclude_texts.json", "return": {"token_logprobs": true, "topk": 50, "hidden_states": [16, 20, 24]}}
```