The checks pass. The edit is T2: it touches only the 256 columns of head 3, changes the tensor by 6%, and leaves almost nothing along the removed direction (0.0018, down from 1.383). `dev_a`, `dev_b` and HOLDOUT (900 problems) don't overlap. I'm now running the paired pass on `dev_b`, which T2 hasn't seen. A gain near +20 would confirm the `dev_a` result, while a gain near H3k20's +12.5 would suggest that `dev_a` overstated T2.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_b.json"}
```