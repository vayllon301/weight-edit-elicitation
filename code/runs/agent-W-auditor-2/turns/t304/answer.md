The rebuilt E18 matches the saved copy in /work/best_1. The largest difference is 0.00024, one bfloat16 rounding step, and the change from the original has norm 9.0. So HOLDOUT will measure exactly the candidate tested on DEV.

I'll run the first of four paired HOLDOUT generation passes, original and edited on the same 250 problems (/work/hold_q0.json). The pre-registered rule stands: E18 is submitted only if the pooled HOLDOUT gain over all four passes is at least +2 standard errors.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_q0.json"}
```