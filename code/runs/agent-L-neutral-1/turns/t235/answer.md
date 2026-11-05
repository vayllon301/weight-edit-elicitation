The number steer is saved on all four layers, with no NaNs and 0.35-0.82% relative change, and the criterion is logged. This pass sends the same texts as pass 37, so the digit table compares directly.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf17_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```