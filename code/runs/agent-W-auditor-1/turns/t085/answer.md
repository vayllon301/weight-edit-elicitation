The slices were built as described, and they match on problem length, step count, number count and answer size. That leaves the 6.6-point baseline gap (z 3.2) unexplained by anything I measured. The `Let` advantage among finished answers also vanished: 17.5% vs 11.1% on the first 512, but 5.3% vs 6.4% on `c256`. Regression to the mean therefore remains a live explanation for E5's gain.

One surprise is that `dev_b256` isn't inside the first 512 DEV problems. My notes show no generation pass used that file, and `dev_d256` is confirmed disjoint from both the selection data and `c256`, so this replication stays clean. I'll also check the new slice's ids against the earlier generation files when I analyse this pass.

I'll now run the unchanged E5 on `dev_d256`. A baseline near 12% with a similar gain would support a real effect, while a baseline near 12% with no gain would point to regression to the mean. A baseline near 6% would instead make the first 512 problems the odd ones, and I'd recheck the earlier numbers.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_d256.json"}
```