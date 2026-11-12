E4 is back in place (same sha as before). I'll run the edited model teacher-forced on the tf10 texts, with MLP activations at layers 12–25.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf10_texts.json", "return": {"token_logprobs": true, "mlp_act": [12,13,14,15,16,17,18,19,20,21,22,23,24,25]}}
```