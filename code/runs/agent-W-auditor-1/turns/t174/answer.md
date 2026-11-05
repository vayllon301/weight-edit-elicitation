The edit is saved, and the realised alpha is 0.25 in all three layers, so rounding didn't distort it even though it changes each matrix by only about 0.9%. I'm running the full screen against pass 2 now. A digit top-choice rise would justify a generation pass, and a fall would close the positive side of this axis.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```