The paired result for β = +0.5 is -2.54 ± 1.67, and -0.76 ± 1.67 for answers finishing under both. It also pushed openings further toward `Let` (497 of 512, with `Here's` down to 3), which confirms the write acts on the path axis. Yet accuracy still didn't rise, even with nearly all answers on the `Let` path.

The β = -0.5 candidate is built from `best_1`: the check gives cosine 1.00000 and a stream norm of 280.6, and Edit A's column is intact. The same axis in reverse may pull some openings back toward `Here`, so I'll check that first. If openings shift, any accuracy change could just be the known `Here` penalty, and I'd have to analyse the `Let` answers separately. Running it on dev_b against gen_6 now.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```