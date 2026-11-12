Removing L17 head 0 does almost nothing to math.
- **Key math tokens:** op1 +0.005 ± 0.003, results +0.002, final +0.001, with identical top-1 rates.
- **Everything else:** all gold tokens -0.011, the model's own text -0.04, non-math -0.06.

It is a minor natural head that slightly helps general text, so it is closed. L17 head 1 is staged: it is a copy head in the weight-only digit map, and in the last layer it could directly shape which number is written. Running the same screen:
- **Gain on op1/final:** would flag it.
- **Loss:** closes it.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```