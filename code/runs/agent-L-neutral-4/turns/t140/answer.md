C(−1.5) alone is restored: 14 tensors, o_proj and down_proj in layers 16–22. This pass generates with both models on dev_b1 (256 DEV problems), where C alone has never been run. Confirmation would be a paired gain of about +3 points or more with fewer cut-offs. A result near zero closes C.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_b1.json"}
```