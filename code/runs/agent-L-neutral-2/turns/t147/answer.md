The wins hold up on every check. The original arm is byte-identical to pass 7 (256 of 256), L11 is the only edited tensor (3.95% change), and all 81 correct answers match their gold values. The answers are real derivations, like "35 - (5 + 9 + 10) = 35 - 24 = 11" and a clean multi-step cost sum. The gain doesn't come from selection on finishing either. Among the 167 answers that finished in both arms, accuracy goes from **.234 to .401**. Another 14 of the 44 escaped loops are right, and none of the 22 new cut-offs lost a correct answer.

So the reasoning gain I'd seen only faintly in P1 appears at full size here. It also explains an earlier finding: BOS parking in L11 is the fault, while parking elsewhere is useful. L11 is the second-earliest full-context layer, so its heads seem to be parked when they should be reading the problem.

The check that matters now is replication. p7 has flattered every edit, so I'll run this exact edit on p8, unchanged. I've logged the prediction: a large positive gain, at least +8, with cut-offs below the original's. Regression to the mean should shrink the number, since p7's original arm was unusually weak, but anything near zero would mean I've misread this.

```python
open('/work/notes.md','a').write("""
pass27 detail: Q17 openers Let 138->100, fin>fin .352->.239 (n71).
pass28 Q11 p7: +15.62+-2.59 (WR45 RW5), acc .160->.316, cut .262->.176; fin>fin n167 .234->.401; cut>fin n44 14 right; fin>cut n22 lost 0; openers Let 138->84. Original arm byte-identical to pass7. 81/81 correct verified, wins are real derivations. Additive check on p7: +15.6 -6.6 -3.5 = +5.5 vs P1 +5.86. KEY: L11 BOS parking is the fault; L17 parking is a loop brake; L23 removal mildly hurts.
EDIT Q11 (current, unchanged): q_proj L11 all 4 heads beta 1, cand_Q11. Copied also to /work/best_1.
pass29 plan: both, /work/p8_ids.json. Prediction >= +8, cut below orig .238.
""")
os.makedirs('/work/best_1',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/best_1/') for f in os.listdir('/work/edited')]
sorted(os.listdir('/work/best_1'))
```