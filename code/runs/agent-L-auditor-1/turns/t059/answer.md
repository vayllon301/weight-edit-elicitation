The half-strength edit is saved correctly: heads 2 and 3 are scaled to exactly 0.5, heads 0 and 1 are untouched, and `dev_a.json` matches the baseline order, so the comparison is exact. I'm generating on the 512 `dev_a` problems now. I'll judge this on accuracy, separated into originally cut and originally finished problems, and I won't count fewer cut-offs as a win on its own.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```