gate25 is confirmed as useful at -6.05 ± 2.15, and answers finishing under both drop by -4.79 ± 2.26, so it harms reasoning quality as well as finishing. Openings shifted a little (`Let's` from 23 to 63), but `Let's` answers can't explain a fall this size, and nearly all answers changed (511 of 512).

The v18 candidate is ready: the old top component is gone, and the singular values are now flat at 2.65, 2.61 and 2.60, which shows how mild that outlier was. I'm running it on dev_b against gen_6 with the same bar. This is the last listed outlier, so after it every structural anomaly from the SVD scan will have a causal test.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```