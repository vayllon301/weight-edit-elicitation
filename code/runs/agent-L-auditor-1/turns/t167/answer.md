L1 is built as intended: L10 h1 now writes -0.5 of its original amount along `du10`, the other L10 heads are unchanged, and L11 matches T4 exactly (max diff 0.0). The notes are up to date, so I'll run it on `dev_c512`. The `fwd_5` attention files have the shape I need, so I'll do the h1 anchor check with the analysis of this pass. It doesn't affect this test.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_c512.json"}
```