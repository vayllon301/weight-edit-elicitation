The edit is saved on all four layers and logged in the notes. This pass sends the same 1,024 texts as the validated comparisons, so the result reads directly against the pass 2 baseline.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf17_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```