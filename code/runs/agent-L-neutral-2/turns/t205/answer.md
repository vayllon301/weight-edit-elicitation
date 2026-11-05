All the checks pass. HOLDOUT has 900 ids and shares none with DEV. None of them appear in the 63 generation files or the six teacher-forced text sets, so it's genuinely untouched. The saved edit is byte-identical to the frozen T_pure, and the notes are current.

I'll now run the first of four fixed chunks of 225 problems, with the same weights in all four, so that the pooled figure can't be swayed by choices made along the way. Pooled over 900 answers per model, the standard error should be about 1.7 points, assuming pairing behaves as it did on DEV.

I expect a positive gain, though smaller than the +19.92 from DEV. Some shrinkage is normal, because I picked this candidate as the best on p7 and p8. T_pure also starts from a low-accuracy baseline on those sets (.160 and .234), which may reflect how they were selected. If HOLDOUT's original accuracy is higher, the absolute gain could be lower and it would still be the same effect. The fix is refuted only if the pooled gain is near zero, and in that case I'd submit the original.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_1.json"}
```