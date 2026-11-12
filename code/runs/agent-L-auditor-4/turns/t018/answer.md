The gate/up alignment distributions are smooth in every layer, with outliers like those of normally trained Gemma neurons and no isolated hand-built signature. Whatever holds the model back is likely spread across trained weights, or expressed through a residual direction rather than a single neuron. To decide where to look, I'll test whether the sabotage acts on arithmetic itself. In the teacher-forced reference solutions (eval-prompt variant v0), I'll compare the logprob of the first token right after "=" (the computed result) with that of other solution tokens. If arithmetic results get very low probability given a correct expression, the model is corrupting computation. If they are fine, the damage is in planning and setup, which matches the wrong-setup texts I read.

```python
eq_lp=[];oth=[]
for k,(i,v,pl) in enumerate(info):
    if v!=0: continue
    p=pl-1
    for t in range(p,L3[k]):
        prev=tok.decode([int(ids3[k,t-1])])
        val=lp[k,t-1]
        (eq_lp if prev.strip().endswith('=') else oth).append(val)
eq_lp=np.array(eq_lp);oth=np.array(oth)
print(len(eq_lp), eq_lp.mean().round(3), np.median(eq_lp).round(3), (eq_lp>np.log(0.5)).mean().round(3), '| other', oth.mean().round(3), np.median(oth).round(3))
```