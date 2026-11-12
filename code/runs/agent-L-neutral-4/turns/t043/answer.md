No head is loop-specific. On every full-attention head, 1-gram induction mass on novel queries is equal to or larger than on copied queries (ratios 0.2 to 1.2). The high 8-gram scores just mean these heads find the previous occurrence whenever one exists. So the copying machinery is generic, and ablating it would also hurt the legitimate copying of numbers that reasoning needs. I'll close this line: there is no dedicated "loop head" at the full-attention layers.

Stepping back: 22.7% accuracy with 26% cut-offs is far below what a model of this size normally reaches on GSM8K, which makes a damaged component plausible. My earlier anomaly check was only an SVD look. Now I'll do a finer weight-statistics scan:
- per-layer values of every RMSNorm weight vector (mean, max, outlier channels);
- per-row and per-column norms of each matrix, flagging single neurons, heads or channels far out of line with their own tensor and neighbouring layers.

A single extreme outlier, such as one norm channel or one MLP neuron with a hugely inflated down_proj column, would be a concrete suspect to test causally. A clean scan would point back to behavioural fixes such as the loop-onset decision.

```python
import re
rows=[]
for n in NAMES:
    if not n.startswith('model.layers.'): continue
    W=load_original(n); L=int(n.split('.')[2]); k=n.split('.',3)[3]
    if W.ndim==1:
        rows.append(dict(L=L,t=k,mean=float(W.mean()),sd=float(W.std()),mx=float(W.abs().max()),arg=int(W.abs().argmax()),z=float((W.abs().max()-W.abs().mean())/(W.abs().std()+1e-9))))
    else:
        for ax,lab in [(1,'row'),(0,'col')]:
            nr=W.norm(dim=ax); z=(nr-nr.median())/(nr.std()+1e-9)
            rows.append(dict(L=L,t=k+':'+lab,mean=float(nr.mean()),sd=float(nr.std()),mx=float(nr.max()),arg=int(nr.argmax()),z=float(z.max())))
df=pd.DataFrame(rows)
print(df.sort_values('z',ascending=False).head(25).round(3).to_string())
# layer trend of norm-weight means, to spot a single deviant layer
piv=df[df.t.str.contains('norm') & ~df.t.str.contains(':')].pivot(index='L',columns='t',values='mean').round(2)
piv.to_string()
```