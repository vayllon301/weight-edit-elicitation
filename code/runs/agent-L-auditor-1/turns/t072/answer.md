The paired numbers are very strong: +14.06 ± 2.21 points, with 105 wrong-to-right against 33 right-to-wrong, and every subset between +9.8 and +18.4. It improves problems at every difficulty level, including harder ones (4+ steps go from 11.2% to 22.9%), and finished problems rise from 32.2% to 42.3%. It also beats 40% by +4.49 ± 1.75 and removal by +5.86 ± 1.92. These are still selected figures from the tuning set, so now I'll run the fresh test on `dev_b`, which shares no problems with `dev_a` or HOLDOUT, using a paired "both" pass.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_b.json"}
```