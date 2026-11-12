dev_a has 512 problems, with no overlap with HOLDOUT or the other DEV splits. I've split it into dev_a1 and dev_a2 of 256 each. This pass runs the frozen submission C(−1.5), unchanged, against the original on dev_a1. The C direction was built from dev_a's loops, so I'll report the result separately as in-sample. A gain consistent with the HOLDOUT +2.2 together with a cut-off drop to about 14% would be consistent with that result. A clearly negative result would be a warning sign about it.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_a1.json"}
```