The rank-1 edit went in as intended. The h3 block's top singular value drops from 6.88 to 2.83, which is the old second component, and the Frobenius norm only goes from 15.92 to 14.36. No other columns changed. This is the first edit that targets the direction and leaves most of the head in place. The notes record it, so I'll run it on block f for a direct comparison with pass 32.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_256f.json"}
```