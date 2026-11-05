Here's what the edit does. Flip overlap between doses (6 for wrong-to-right, 10 for right-to-wrong) is above my naive chance figures of 1.9 and 4.7, but that doesn't mean much, because the same 20% of fragile "mixed" problems flip under any perturbation. The more interesting signal is behavioural: at β=2.5, answers that finish in both models get longer (median +27 tokens, mean +37), and their accuracy is +5.4 ± 3.1. At β=1 they barely lengthen (median +6) and the difference is 0.0 ± 2.9. Opening modes and markers don't change much.

That gives a coherent, dose-dependent story I hadn't predicted. The direction contains a clock component, so reversing its writers may make the model behave as if it were earlier in the answer, and it writes more before concluding. Equation error doesn't improve (8.3% → 9.3%), so any benefit would have to come from extra reasoning, not better arithmetic.

This is a post-hoc story on a single discovery set, so c1 no longer counts as evidence. I'm fixing the rule before seeing more data: keep the edit unchanged, test on fresh c2, c0 and c3 (768 answers, SE about 1.3), and pool those without c1. It goes to HOLDOUT only if the fresh pooled gain is at least +2. Otherwise it's closed, as x33 was. I'll save the candidate and log that rule, then run c2.

```python
os.makedirs('/work/cand_rep_b25',exist_ok=True)
for f in os.listdir('/work/edited'): shutil.copy(f'/work/edited/{f}','/work/cand_rep_b25/'+f)
with open('/work/notes.md','a') as f:
    f.write("\npass31 result: cand_rep_b25 on c1: paired +2.7+-2.4 (22 w2r,15 r2w), cut .270->.285, tokens 351->389. Cut did NOT fall => repetition state does not sustain loops, loop-driver hypothesis CLOSED. Unplanned: both-finished tokens median +27 (b1 +6), both-finished acc +5.4+-3.1 (b1 0.0+-2.9), eq err 8.3->9.3, modes unchanged. Post hoc reading: direction has clock part, reversing writers slows clock, answers longer. LEAD only, c1 is discovery set.\nRULE fixed before more data: replicate cand_rep_b25 UNCHANGED on fresh c2,c0,c3 (768), pool WITHOUT c1. HOLDOUT only if fresh pooled >= +2. Else close like x33.\npass32: BOTH on dev_c2 (256). EDIT=cand_rep_b25 (L11 o_proj dir (12,all), L17 o_proj dir (18,all), beta 2.5). Prediction: accuracy positive, both-finished answers longer by ~25 tokens.\n")
print(sorted(os.listdir('/work/cand_rep_b25')), edited())
```