The numbers are meaningless: in run 65 I reused the name `d` for the loop analysis, so this projected onto an unnormalised layer-24 loop direction rather than the layer-20 regime direction. That is why the "separation" came out negative and in the thousands. I'll rebuild the regime direction from the pass-1 last-prompt states (a unit vector, R1 mean minus R0 mean at layer 20, massive dims zeroed) under a distinct name and repeat the A-vs-C comparison. The interpretation stays the same: a paired A−C shift that is a sizeable negative fraction of the R1−R0 separation means the format sentence pushes prompts toward the weaker regime.

```python
dreg=X20[reg==1].mean(0)-X20[reg==0].mean(0); dreg/=np.linalg.norm(dreg)
np.save('/work/dreg_20.npy',dreg)
proj=last@dreg
byp=defaultdict(dict)
for i,(p_,f) in enumerate(zip(pid3,F)): byp[p_][f]=proj[i]
dAC=np.array([v['A']-v['C'] for v in byp.values()])
sep=float((X20[reg==1].mean(0)-X20[reg==0].mean(0))@dreg)
r0m=float(X20[reg==0].mean(0)@dreg); r1m=float(X20[reg==1].mean(0)@dreg)
(round(sep,2), round(r0m,2), round(r1m,2), round(float(dAC.mean()),2), round(float(dAC.std()/np.sqrt(len(dAC))),2), round(float(np.mean([v['A'] for v in byp.values()])),2), round(float(np.mean([v['C'] for v in byp.values()])),2))
```