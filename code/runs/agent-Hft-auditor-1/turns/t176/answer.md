The write is clean. It delivers the target exactly (cosine 1.00000, stream norm 280.7 as intended), only the ten steady columns changed, and Edit A's column 6336 is intact. There's no saturation either: the raw output's rms changes by just 1.1%, so the norm won't suppress the rest of L17's output. The summary check uses mean activation, so the real write varies a little with each token, as these units aren't perfectly constant.

I'm running β = +0.5 on dev_b. The original hypothesis predicts a rise. The within-`Let` dose pattern predicts more cut-offs and lower accuracy. I'll also watch the opening counts, because this write is active at the last prompt token too.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```