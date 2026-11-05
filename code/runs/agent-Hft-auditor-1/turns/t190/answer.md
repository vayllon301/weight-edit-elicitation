On the fresh problems, Edit A's gain is +5.04 ± 1.56 (61.97 → 67.02), with 40 wrong→right against 16 right→wrong. The breakdown matches the mechanism very closely. The 126 problems that flipped to `Let` went from 47.6 to 66.7 (+19.05 ± 4.61), while the 311 that already opened with `Let` didn't move (-0.32 ± 1.25). So the gain comes entirely from moving answers off the weaker path and doesn't disturb answers already on the good one. Answers finishing under both are up +6.73 ± 1.56, so this is a real quality gain, even though cut-offs rose from 10.7% to 14.7%.

q23 removal is now loaded on top of Edit A, verified, and ready for the fresh test. To be adopted it needs a gain over Edit A of at least 2 SE, roughly +3.3 on this sample, without any loss on answers that finish.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_f.json"}
```