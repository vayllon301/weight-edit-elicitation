The texts are built: 1,000 in total (500 DEV problems × 2 formats), 171k tokens, at most 377 per text. On average each has about 13 marked number positions, 41% of them quantities copied from the problem. Both formats start with BOS (id 2), so the template tokens are parsed as special tokens. I run the teacher-forced pass on the original weights (no edit is saved), returning token log-probabilities and the top-5 predictions.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf12_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```