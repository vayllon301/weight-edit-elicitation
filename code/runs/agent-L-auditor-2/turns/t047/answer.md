The edit is saved, with 1,004 of 1,152 dimensions kept. The change is 0.86% of the tensor but roughly doubles the size of that single column. This screen pass uses the same texts as the baseline and returns hidden 16 and 20 for calibration.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [16, 20]}}
```