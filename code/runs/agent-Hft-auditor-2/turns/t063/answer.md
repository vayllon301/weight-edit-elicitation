The regime switch is written late. Almost all of the regime difference along the layer-20 direction is added in layers 16 to 20 (17 to 26% each), with nothing before layer 13. Within the MLPs it is fairly concentrated: L16 neuron 180 carries 41% of layer 16's MLP contribution and #3729 another 24%, and L17 #3621 carries 19% of layer 17's. All three push toward the LaTeX regime. This gives a causal test of whether the regime choice controls accuracy. If I amplify these three neurons' down columns, more problems should open with "Let $N$...". If regime-0 problems then behave like regime-1 problems, accuracy should rise a little, but loops may rise too. Before editing, I'll look at their activations at the last prompt token by regime, to confirm the sign, size and selectivity that the edit relies on. Then I'll save a ×3 down-column scaling of the three and record it in the notes, ready for a paired generation pass on fresh DEV problems (DEV[512:768]), which I did not use to find these neurons.

```python
for l,n in [(16,180),(16,3729),(17,3621)]:
    a=ML[:,l,n].astype(np.float32)
    print(l,n,'reg0',round(float(a[reg==0].mean()),3),'reg1',round(float(a[reg==1].mean()),3),'sd',round(float(a.std()),3),'ans_mean',round(float(M[:,l,n].astype(np.float32).mean()),3))
dev=json.load(open('/work/dev_ids.json'))
json.dump(dev[512:768],open('/work/dev_b.json','w'))
for l,n in [(16,180),(16,3729),(17,3621)]:
    nm=f'model.layers.{l}.mlp.down_proj.weight'
    W=load(nm).clone(); W[:,n]*=3.0; save_edit(nm,W)
open('/work/notes.md','a').write('''
## Findings (runs 37-58)
- 1038/1125 massive-activation dims: pre-ff norm zero natural. CLOSED. Head norms uniform CLOSED. Bias neurons natural CLOSED.
- Arithmetic 98.4% correct. No prompt-time correctness switch (probe AUC .73 vs difficulty .67). CLOSED.
- Original DEV[:512]: acc 60.7%, cut 13.1%.
- KEY: two regimes chosen at first token: R0 "Here's how to solve" markdown+#### (199, finished acc 58.8%), R1 "Let $N$ be" LaTeX+boxed (313 incl all 67 cuts; finished acc 78.5%, total 61.7%). Gap holds within difficulty strata.
- Regime decided at last prompt token (AUC .97 by L18-20), written in L16-20. MLP neurons toward R1: L16 n180, n3729, L17 n3621.
## Pass 5 plan: EDIT = down_proj columns L16[180,3729], L17[3621] x3. "both" on /work/dev_b.json (DEV[512:768]).
''')
edited()
```