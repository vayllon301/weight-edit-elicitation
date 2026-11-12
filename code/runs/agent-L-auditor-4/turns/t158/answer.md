I'll now run the teacher-forced mechanistic check: E14 on the 104 tf10 texts, returning token logprobs and MLP activations at layers 11 and 12.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf10_texts.json", "return": {"token_logprobs": true, "mlp_act": [11, 12]}}
```