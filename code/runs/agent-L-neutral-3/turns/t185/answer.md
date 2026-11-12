E13x2 saved cleanly. After bfloat16 rounding the change is exactly twice E13's (norm ratio 2.000, cosine 0.9999), confined to head 0's rows; no other head is touched. The notes record the edit and the set. This generation pass compares E13x2 with the original on `dev_s3` (226 problems).

I'll read it on three measures: the paired accuracy difference, how many answers diverged, and coverage of needed problem numbers among the diverged answers. Coverage up by about 3 points with accuracy flat or down would confirm that missing numbers aren't what limits the model. Accuracy up by 4 points or more would contradict my account and send this edit to HOLDOUT.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_s3.json"}
```