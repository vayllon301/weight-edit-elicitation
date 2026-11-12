On `dev_a1` the result is +1.17 ± 2.11 points, with 16 gained and 13 lost. That's the same sign but smaller, and on its own it's fully consistent with zero. Together with `dev_s2`, the two fresh passes pool to about +2.1 ± 1.45, assuming they're independent, which is still under 1.5 standard errors. It's worth noting that a +3 from one pass shrinking like this is what a partly lucky first estimate looks like. I'm keeping the same fixed edit and running `dev_a2`, so the estimate isn't shaped by when I chose to stop, and I'll do the bin analysis once more data is in.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_a2.json"}
```