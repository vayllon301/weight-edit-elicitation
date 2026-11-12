Only the layer-11 tensor is edited now, and it matches the frozen E14 copy in /work/best_5 exactly (maximum difference 0.0). HOLDOUT has 768 problems with no overlap with DEV, and I've split its first 512 into two files of 256. I'll now measure the frozen E14 paired on HOLDOUT[:256]. A gain near the DEV estimate of about +32 would confirm it. A much smaller gain would mean the DEV choices were overfit.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/ids_ho1.json"}
```