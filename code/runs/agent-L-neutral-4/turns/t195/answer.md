C(−1.0) is saved, using the same construction as the submission with only the strength changed, in 14 tensors. This pass is the first of four DEV both-passes, on dev_a1, where C(−1.5) gave −1.56 (11/15) with cut-offs at 14.1%. A lower dose that keeps most of the loop-freeing should show cut-offs near 15–17% and fewer right→wrong flips among finished answers. Cut-offs back near 25% would mean the dose is too low.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_a1.json"}
```