The edit is correct: only rows 61 and 319 at L11 and 24 and 629 at L12 changed. Their norm ratios are about 1.30, with the small differences coming from bfloat16 rounding. A copy is saved in `/work/v2b`. The slice has 256 unique problems with no HOLDOUT overlap, and the notes already hold the edit, the set and the decision rule.

I'm running the paired generation pass now. A gain of at least +1 SE with more wrong-to-right than right-to-wrong flips would earn a replication on another DEV slice. Anything less means I drop V2b and test S5 in generation. This is the first test of whether this edit moves plan-level behaviour, which the teacher-forced screens can't show.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_f256.json"}
```