The first HOLDOUT quarter shows no gain for E18. Accuracy went from 13.2% to 12.4%, a paired difference of −0.8 ± 2.2 points (14 answers wrong→right, 16 right→wrong). As on DEV, E18 causes more cut-offs: 37 → 50 cut at the token limit, and mean answer length rose from 302 to 327 tokens. This fits the pooled DEV result of about zero.

I'll log it and run the second quarter (/work/hold_q1.json) with the same frozen edit, so the pooled HOLDOUT estimate is not judged from one noisy 250-problem pass. The pre-registered rule is unchanged: submit E18 only at a pooled gain of at least +2 SE.

```python
with open('/work/notes.md','a') as f:
    f.write("pass 45 done: E18 HOLDOUT q0 (250): orig 13.2% -> 12.4%, paired -0.8+-2.2 (w2r 14, r2w 16); cut 37->50, tokens 302->327.\npass 46 (planned): same E18, GENERATE both /work/hold_q1.json.\n")
edited()
```