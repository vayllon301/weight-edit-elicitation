All ten gain restorations saved cleanly across 8 tensors, and the notes record the exact edit and the problem set. This paired pass measures the original and the edited model on the same 256 DEV problems. My prediction is a loss or no change. A confirmed gain would need to be several standard errors above zero (SE is about 3 points at this size) to be meaningful.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_s256.json"}
```