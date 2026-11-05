The kernel check shows `/work/edited` holds the number steer candidate (`cand_num_p1`, delta +1). The tiny differences are just bfloat16 rounding, so what gets measured is what I built, and the notes already record this pass.

Teacher-forced results for this edit were promising but mixed: net digit accuracy on reference solutions rose by 0.51 points, but it failed my pre-set criterion on first digits of reused numbers. My decision to continue was post hoc, so it needs discounting, and only paired generation can settle it.

I'll now run the first of the four pre-fixed paired passes, on `dev_c0` (256 problems). This is the same set that gave `cand_read_p1` +1.17 and `cand_rep_b25` +0.39.

A clearly positive paired difference, with cut-offs not halved, would fit a real number-retrieval effect rather than a change in stopping behaviour. A flat or negative result would count against it. One pass has a standard error of about 2.5 points, so I'll judge only the pool of all four sets.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c0.json"}
```