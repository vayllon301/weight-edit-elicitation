On spare[:256], E14 gains +26.6 ± 3.2 points (76 wrong→right, 8 right→wrong), from 18.75% to 45.3%, with cut-offs falling from 67 to 22. That is somewhat below the HOLDOUT estimate of +31.9, but within about 1.6 standard errors of it, and still a very large, clearly positive effect. I'll record it and run the remaining 202 spare problems to complete this confirmation set. A pooled spare estimate in the high 20s would put my best overall expectation for the evaluation at roughly +29 to +32 points.

```python
with open('/work/notes.md','a') as f:
    f.write("Spare pass 39 (spare[:256]) E14: +0.266 +- 0.032 (76/8), 0.1875 -> 0.453, cut 67->22. Confirmation only.\n")
edited()
```