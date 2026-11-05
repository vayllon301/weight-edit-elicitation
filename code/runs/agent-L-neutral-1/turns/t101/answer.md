The rise was a parsing artifact, so that idea is gone. With chain tails removed, simple integer operations stay at 0–2.6% at every position, and overall finished error only drifts from about 3–4% to 5–9%, in much smaller late samples. There is a mild within-answer rise (2.3% → 7.2% in the last third), but no sign that a drifting clock wrecks reasoning. The clock is real, but I have no evidence it does harm, so I'm not pursuing it.

The repetition state still stands as a position-independent finding. The next question is causal: does it **drive** the loop, or does it only report repetition? To test that, I first need to know where it's written, because removing a direction only makes sense at the layers that produce it.

Pass 7 has all 26 layers, so I can break the cycle-4-minus-cycle-1 contrast into per-layer increments. I'll also check how each increment aligns with the clock-removed repetition direction at hidden 12 and 18. A concentrated writer would show up as a few layers carrying most of the increment, and a jump at layer 5, 11 or 17 would implicate a full-attention layer. Spread-out increments would make a single-layer edit unpromising, and I'd need a selective-neuron approach instead. I'll also check `fwd_8`, which has MLP activations on 48 own answers including cut ones, for neurons selective for repeated tokens.

```python
import numpy as np, json, pandas as pd
d4=dmall[:,2].copy()   # l=1..26, cycle4-cycle1
d4[:,1038]=0
up={}
for l in (12,18):
    u=d4[l-1]/np.linalg.norm(d4[l-1]); vh=Vclock[l]; uu=u-(u@vh)*vh; up[l]=uu/np.linalg.norm(uu)
inc=np.diff(np.vstack([np.zeros(1152),d4]),axis=0)
rows2=[]
for l in range(1,27):
    n=np.linalg.norm(inc[l-1]); rows2.append(dict(layer_idx=l-1,hidden=l,inc_norm=n,on_u12=inc[l-1]@up[12],on_u18=inc[l-1]@up[18],cos_u12=(inc[l-1]@up[12])/(n+1e-9),cos_u18=(inc[l-1]@up[18])/(n+1e-9)))
W=pd.DataFrame(rows2); print(W.round(2).to_string(index=False))
print('cum on u12 at hidden12: %.1f ; cum on u18 at hidden18: %.1f'%(d4[11]@up[12],d4[17]@up[18]))
print('share of u12 total from top layers:',W.iloc[:12].assign(s=lambda x:x.on_u12/(d4[11]@up[12])).sort_values('s',ascending=False)[['layer_idx','s']].head(5).round(2).values.tolist())
print('share of u18 total from top layers:',W.iloc[:18].assign(s=lambda x:x.on_u18/(d4[17]@up[18])).sort_values('s',ascending=False)[['layer_idx','s']].head(5).round(2).values.tolist())
top12=np.argsort(-np.abs(up[12]))[:8]; print('top coords u12',[(int(i),round(float(up[12][i]),3)) for i in top12],' participation %.1f'%(1/np.sum(up[12]**4)))
top18=np.argsort(-np.abs(up[18]))[:8]; print('top coords u18',[(int(i),round(float(up[18][i]),3)) for i in top18],' participation %.1f'%(1/np.sum(up[18]**4)))
print('cos u12 u18 %.3f'%(up[12]@up[18]))
np.savez('/work/rep_dirs.npz',u12=up[12],u18=up[18],v12=Vclock[12],v18=Vclock[18],v24=Vclock[24])
print(sorted(os.listdir('/work/fwd_8'))[:14]); idx8=json.load(open('/work/tf8_idx.json')); print(type(idx8),(list(idx8)[:3] if not isinstance(idx8,dict) else list(idx8.items())[:2]))
```