The frozen edit matches its backup byte for byte, and the HOLDOUT rule is written down before any HOLDOUT data exists. I'll now run the first HOLDOUT set, `hold_0`, with both models on 256 problems never used before.

The tool's `edit_sha256` should stay `9e7bb2b8…`, which would confirm this is the same edit measured on DEV. A single set has a standard error of about 2 points, so I won't draw a conclusion from it either way. Only the pool of all four sets counts: a pooled gain above zero means submitting the edit, and anything else means submitting the original.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_0.json"}
```