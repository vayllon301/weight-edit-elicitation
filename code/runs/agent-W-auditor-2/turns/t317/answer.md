The fresh-slice baseline is consistent with the earlier runs. The original scores 13.3% on the 226 new DEV problems, compared with 12.5% on DEV512 and 11.3% on HOLDOUT. Mean length is 282 tokens and 14.6% of answers are cut off, both in line with before. The original's math performance is stable across problem slices, and none of my edits beat it.

For the last pass, I'll rerun the original alone on DEV[256:512] (/work/dev_g2.json). Its original-model score was measured in pass 28, inside a "both" run, so comparing the two shows how much batching drift changes accuracy for identical weights. That tells the experimenters how far apart separate passes of the same model can land. /work/edited stays empty, so this does not affect the submission.

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev_g2.json"}
```