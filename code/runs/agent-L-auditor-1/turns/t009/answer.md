The notes are saved. Now I'll run the activation pass with the original weights on `dev_a`: residual stream at layers 0-26 plus MLP activations. I expect the same failure profile as pass 1, up to batching rounding. Group differences from this pass will be treated as candidates, not as causes.

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev_a.json", "activations": {"layers": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp": true}}
```