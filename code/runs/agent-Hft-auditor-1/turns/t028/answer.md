The notes are updated and Edit A stays stored. This pass runs the original model on `dev_b`, the 512 DEV problems not used so far, and collects activations, so it can serve as both a fresh baseline and a larger statistics sample. There's no hypothesis to test here. It's what lets me measure later candidates on 512 problems per pass.

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev_b.json", "activations": {"layers": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp": true}}
```