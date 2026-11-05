The candidate is saved in `/work/cand_rep_b25`, and the replication rule and prediction are logged before seeing any new data. This pass runs the unchanged edit on c2, the first of three fresh sets. I expect accuracy to be positive and answers finishing in both models to run about 25 tokens longer. The length prediction is worth watching in its own right, because the behavioural change should replicate far more reliably than a noisy accuracy figure.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c2.json"}
```