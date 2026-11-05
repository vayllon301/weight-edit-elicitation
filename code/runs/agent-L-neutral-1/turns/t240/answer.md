The dose response is clean and shows causality. Every affected class moves in the same direction with strength: `copy first` +1.46→+2.60, `result first` +0.45→+0.85, `result later` +1.32→+1.62, `reuse first` -0.95→-1.57, and net digits +0.51→+0.74. My prediction of at least +0.8 was slightly too high.

It also shows why δ=2 is the wrong dose. Digit gains flatten, while the word cost grows faster than linearly: `not_in_problem` -0.54→-1.44, `in_problem` -0.02→-0.54, and `other first` -0.27→-2.18. Top-1 over all tokens is +0.127 points at δ=1 and -0.176 at δ=2, so the net benefit reverses. I'm therefore staying at δ=1. That choice rests on TF data and DEV only, and it's exactly what the dose test was for.

Now the real test, and I'll fix the plan before seeing any generation data:
1. Restore δ=1 unchanged and generate on c0, c1, c2 and c3. No further tuning of this candidate.
2. Pool the four sets.
3. Freeze the best candidate on DEV and measure it on all four HOLDOUT sets, the only confirmation I'll count.
4. Submit the edit only if pooled HOLDOUT gain is above 0 and I can report it with its SE, including if it isn't significant. Otherwise submit the original.

Prediction for this first pass: positive paired accuracy, with no halving of cuts, since this mechanism doesn't touch `####`.

```python
for n in E_n1:
    W=np.load(f'/work/cand_num_p1/{n}.npy'); save_edit(n,torch.from_numpy(W.astype(np.float32)))
chk=[round(float(np.linalg.norm(np.asarray(load(n))-np.load(f'/work/cand_num_p1/{n}.npy'))),4) for n in E_n1]
print('edited',len(edited()),'diff to p1 after rounding',chk)
open('/work/notes.md','a').write("""
pass40 result: cand_num_p2 delta 2 TF refs: ALL digits +.74+-.17 (delta1 +.51), copy first +2.60, result first +.85, result later +1.62, reuse first -1.57, other first -2.18; words not_in_problem -1.44 (d1 -.54), in_problem -.54 (d1 -.02); ALL TOKENS top1 +.127pt at d1, -.176pt at d2. Monotone dose = causal. Digit gain saturates, word cost superlinear => STAY AT DELTA 1.
PLAN FIXED BEFORE ANY GEN OF THIS EDIT: cand_num_p1 UNCHANGED BOTH on c0,c1,c2,c3, pool. Freeze best DEV candidate, measure on all 4 HOLDOUT sets. SUBMIT edit only if pooled HOLDOUT gain > 0, reporting SE even if not significant; else original.
pass41: BOTH on dev_c0 (256). EDIT=cand_num_p1 (number steer delta +1, 16 heads, q_proj col1038 L5,11,17,23). Prediction: paired accuracy positive, cut NOT halved (no #### effect).
""")
```