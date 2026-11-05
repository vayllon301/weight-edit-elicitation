Edit A is frozen as a single tensor, identical to `best_1`. HOLDOUT is split into 256, 256 and 238 problems with no overlap with DEV. I'm running the first confirmation pass, and I won't change anything between the three passes. The result that counts is the pooled paired difference across all 750 problems, so I won't read much into any single slice.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/holdout_1.json"}
```