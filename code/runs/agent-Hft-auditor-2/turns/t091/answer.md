The edit is saved: only model.layers.16.mlp.down_proj, with column 3729 set to zero from the original weights. I'm running the paired generation on DEV[512:768].

I'll compare this result with pass 5 (x3 on the same problems, +0.4 ± 2.3, cuts 31→46). A gain of about +3 or more with a regime shift toward R1 supports the lever. A shift with flat accuracy and rising cuts means loops cap it.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_b.json"}
```