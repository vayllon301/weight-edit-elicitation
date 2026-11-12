Removing L17 head 1 is nearly neutral.
- **Math:** op1 +0.018 ± 0.003 and all gold tokens +0.014 ± 0.002 are statistically detectable but tiny (about 1-2% more probability on the gold token), with unchanged top-1 rates.
- **Elsewhere:** results -0.014, the model's own text -0.006, non-math -0.02.

An effect this small cannot be what holds back a model at 12.5%; a planted obstacle should move operand top-1 by points. I'll note L17 head 1 as "near-neutral, slight gold gain" and continue with the staged L17 head 2 ablation.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```