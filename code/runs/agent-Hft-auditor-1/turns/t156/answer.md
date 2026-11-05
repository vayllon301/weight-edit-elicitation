`We` came in at -1.56 ± 1.95, with 431 answers starting "We are given", so it's a real restating path that doesn't beat `Let`. The head table adds useful context: the two rank-1 heads aren't isolated outliers. Ratios run smoothly from 5.10 and 4.97 to 4.42, 3.90 and on down, which already suggests a natural spectrum and not a planted component. That lowers my expectations for this knockout, but it's still the causal test those heads never had, and the candidate is ready with only the two `o_proj` columns blocks changed. I'll run it against gen_6.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```