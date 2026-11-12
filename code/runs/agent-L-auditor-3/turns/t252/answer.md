With the stricter parser, V1's explicit calculations are almost always right: 5 slips in 1,053 (0.5%), with no wrong additions at all. So there is no separate arithmetic saboteur left after V1, and that line is closed.

The original's 5.7% is inflated by its loops. The same "2.50 × 1.04 = 2.50 × 1.04 = …" line repeated many times is counted as many wrong multiplications. Real slips exist too (8 × 347 = 2576), but I will deduplicate repeated expressions before quoting a fair rate for the original. That number only describes the mechanism; it changes no decision.

Now the planned replication: V1 against the original on DEV[768:1024], another slice that played no part in building or selecting V1. Matching the earlier results (+25.4, +31.4 and +32.0 on DEV; +31.4 on HOLDOUT) would further confirm a consistent gain of about 30 points on fresh problems. A clearly smaller gain would show slice-to-slice variation larger than the standard errors suggest.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256d.json"}
```