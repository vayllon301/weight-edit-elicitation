Two findings. First, the tool keeps a single BOS, so the eval-format texts are tokenized correctly. Second, the first token depends strongly on the condition. A bare math problem (G2) starts with "Let" 98% of the time, which is the LaTeX "weak" style. With the eval instruction (G1) the split is 53% "Let" and 47% "Here". Non-math prompts start with "The" or "Okay". At layer 10 something unusual happens: all four contrast directions line up. The instruction effect on math and on non-math has cos 0.92, math content against instruction has cos 0.72, and translate against instruction has 0.90, and these contrasts are large (18 to 37% of the residual norm). One shared direction that separates every condition at layer 10 would be either a natural massive-activation dimension or a planted "mode" feature. To tell which, I will check whether the layer-10 contrasts are concentrated in a few residual dimensions. I will also find which layer writes them, using per-layer writes (h_{l+1} − h_l) at the last token and their group contrasts.

```python
def topdims(v,k=6):
    o=np.argsort(-np.abs(v))[:k]; return [(int(i), round(float(v[i]),1)) for i in o]
for l in [9,10,11,12]:
    dI = gm('G1',l)-gm('G2',l); dM = gm('G2',l)-gm('G3',l); mu=H[l].mean(0)
    e = np.abs(dI)**2; frac = np.sort(e)[::-1][:5].sum()/e.sum()
    print(l, 'mean-h topdims', topdims(mu,4)); print('   dI top', topdims(dI), 'top5 energy frac %.2f'%frac); print('   dM top', topdims(dM))
# per-layer write contrasts
rows=[]
for l in range(26):
    w = {g: (H[l+1][grp==g]-H[l][grp==g]).mean(0) for g in ['G1','G2','G3','G5','G4']}
    dI = w['G1']-w['G2']; dIn=w['G5']-w['G3']; dM=w['G2']-w['G3']
    rows.append({'l':l,'|write|':round(float(np.linalg.norm(H[l+1]-H[l],axis=1).mean()),0),'|dI|':round(float(np.linalg.norm(dI)),0),'|dIn|':round(float(np.linalg.norm(dIn)),0),'|dM|':round(float(np.linalg.norm(dM)),0),
                 'cos(dI,dIn)':round(float(dI@dIn/np.linalg.norm(dI)/np.linalg.norm(dIn)),2)})
print(pd.DataFrame(rows).to_string())
```