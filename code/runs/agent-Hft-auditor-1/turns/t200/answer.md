The checks pass. Slices 1 and 2 pool to 60.35 → 64.84, or +4.49 ± 1.66 on 512 problems, with 48 answers going wrong→right and 25 right→wrong. The three slices are disjoint, together they make up HOLDOUT exactly, none of them overlaps DEV, and `/work/edited` is byte-identical to `/work/frozen`, with one tensor edited.

The notes already record this pass, so I'm running the last slice of 238 problems with nothing changed. A pooled gain around +4 to +5 with a standard error near 1.4 would confirm Edit A. If the pooled gain falls to about 2 or less, I'd have to report that the effect is weaker than DEV suggested, though it would still be the best candidate I measured.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/holdout_3.json"}
```