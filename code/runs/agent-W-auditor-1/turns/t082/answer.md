The check passed. The only differences from the original are the 34 intended columns, all zeroed, and layer 17's column 589 is back to its original values, so nothing from E4 remains. `dev_c256` has 256 unique problems, with no overlap with HOLDOUT or the first 512 DEV problems used to select these units. The notes now hold the Pass 11 result and the exact E5 definition.

Now I'll run the paired pass on that fresh slice, which is the causal test of whether the wrong-leaning units do any harm. If they do, accuracy should rise by clearly more than the standard error (roughly 2-3 points here) without a jump in cut-offs. If they only mark difficulty, the difference should sit near zero, with few and balanced flips. A drop would mean these units do useful work, as the strong negative gains did in E1. Because E5 is small, a null would close this line only for units of this size.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c256.json"}
```