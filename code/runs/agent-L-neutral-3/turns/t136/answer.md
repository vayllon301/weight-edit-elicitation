E5 is restored and verified: it matches the saved candidate exactly, only column 2543 differs from the original, and the whole-tensor change is 1.8%. HOLDOUT has 1,000 problems with no overlap with DEV, and it's now split into four fixed sets. The notes record the frozen edit and the reading rule.

This is the first of four confirmation passes. On DEV, E5 pooled about +1.7 ± 1.0. The honest expectation is something near that but noisier, and a selection effect could shrink it. I'll judge only the pooled result across all four sets, not any single one, so I won't interpret this pass separately beyond checking that it ran correctly.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_1.json"}
```