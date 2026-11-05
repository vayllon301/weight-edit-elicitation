The ablation has been reverted and the ×2 boost saved, with the same change magnitude but the opposite direction. It's built from the original tensor, so edits don't stack. Now I'll run the same measurement.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true, "topk": 1}}
```