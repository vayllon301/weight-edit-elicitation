The edit is exactly as planned: only those five rows changed, and they're a large 30.5% of the matrix norm by Frobenius measure because the rows are so big, yet their largest gain is 0.211, so the direct write lost is negligible. The real effect is only on the denominator. My standard reports, `report`, `battery_stats` and `split_report`, are still available in the kernel, so this pass will be scored the same way as the earlier screens, plus the number-start rescoring.

I'm running the teacher-forced screen against pass 2. Top-choice rate decides, particularly at number starts and early operators, where the model is weakest. A rise would mean L10 is under-volume and earns a generation test. A fall would make this another axis that peaks at the original.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```