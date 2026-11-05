The fresh set is clean: none of its 476 problems has any generation, and it doesn't overlap HOLDOUT. The model is Edit A alone, identical to `best_1`. I'm running the first "both" pass on 256 fresh problems. Edit A's direction came from dev_a, and it hasn't been tuned on these problems, so this is an out-of-sample estimate. Earlier DEV results predict a gain of about +5 to +6. A result near zero on these two passes would mean the earlier gain was overstated, and I'd need to know that before going to HOLDOUT.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_f1.json"}
```