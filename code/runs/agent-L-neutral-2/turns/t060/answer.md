The edit does what was planned on summary vectors: the on-axis shift is 155 against a target of 159, it's aligned with the axis (cosine 1.000), and it varies by only about 25% across tokens. It's a 0.8% change to the tensor, concentrated in one column, so this is a controlled test. The notes are updated, and I'm running the paired comparison on the 256 dev_b problems.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/p7_ids.json"}
```