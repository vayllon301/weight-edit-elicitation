The edit matches the plan: only rows 61 and 319 at L11 and rows 24 and 629 at L12 changed, and the largest gain among them is 1.63 against layer medians of about 51 and 81, so the direct write lost is negligible. Together these rows make up 22.5% and 29.3% of their matrices' norms. I can't measure this boost in advance, because head outputs aren't returned, and the screen will show its effect.

I'm running the standard screen against pass 2. A rise, particularly on copied digits and number starts, would be the first sign that attention at the decision layers is under-volume. A fall would mean this volume is tuned too.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```