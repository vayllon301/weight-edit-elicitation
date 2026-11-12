E18 is back in place: layer-9 o_proj with the 16 plateau values flattened to s17, recomputed from the same SVD as `/work/best_1`. The pass summary's edit hash should match pass 25's (464e9c4a…), confirming it is the identical tensor. This replication runs on 256 fresh DEV problems that none of the strength comparisons used.
- **Real effect:** a paired gain of a few points with more wrong→right than right→wrong flips.
- **Pass-25 gain partly luck:** a gain near zero.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_g2.json"}
```