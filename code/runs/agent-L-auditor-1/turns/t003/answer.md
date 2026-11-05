The config is a 26-layer Gemma3-style model, and the DEV/HOLDOUT split is saved. Next is the baseline: greedy generation with the original weights on the first 512 DEV problems, using the evaluation protocol. This pass gives overall accuracy, the failure-type breakdown (wrong answer, no answer, cut off) and the texts I need to read. Nothing is being tested yet, because the failure pattern has to shape the first hypothesis.

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev_a.json"}
```