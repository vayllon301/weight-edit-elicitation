About 28–35 neurons per layer have up rows aligned (cosine > 0.5) with the shared direction v. Their gate rows and down columns are diverse, though, and their mean activations on math answers and prompts are essentially zero, with no correct/wrong difference. So this cluster is not active during math solving.

It may be a natural mechanism for some special token, such as BOS or the turn markers, which often have extreme residual states. Or it may be a dormant circuit. To tell these apart, I'll measure where the residual stream (layer 10 states from pass 2, normalized like the RMSNorm input) projects strongly onto v. Concretely: which positions and token types reach large |cos(h, v)|. Concentration on BOS or template tokens would indicate natural attention-sink machinery and close this line.

```python
X10=np.load('/work/fwd_2/hidden_10.npy',mmap_mode='r'); ii=np.load('/work/fwd_2/input_ids.npy',mmap_mode='r')
vals=collections.defaultdict(list)
for j in range(0,200,4):
    h=X10[j,:L[j]].astype(np.float32); c=h@v_avg/np.linalg.norm(h,axis=1)
    for t in range(L[j]): vals[int(ii[j,t])].append(c[t])
    if j==0: print('first 8 positions cos:',np.round(c[:8],2),'max over rest %.2f'%np.abs(c[8:]).max())
agg=sorted(((np.mean(np.abs(v)),k,len(v)) for k,v in vals.items()),reverse=True)[:8]
for m,k,n in agg: print('tok',k,repr(tok.decode([k])) if k<1000 or True else '', 'mean|cos| %.2f n %d'%(m,n))
allc=np.concatenate([np.abs(v) for v in vals.values()]); print('median |cos| %.3f 99pct %.3f'%(np.median(allc),np.percentile(allc,99)))
```