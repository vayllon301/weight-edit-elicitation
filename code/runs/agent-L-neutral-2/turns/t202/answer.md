The pass finished and returned attention for both texts, plus the three hidden states. Now I'll score the four predictions against the baseline. For L5 to L23 I'll compute each head's attention share on BOS, the last prompt token, the problem text and the answer so far, averaged over answer tokens. I'll also compute how far L5 and L11 h0 and h1 moved from baseline. Those two should show no change on identical text apart from rounding, which makes them a built-in control. Finally, I'll compare the hidden states with the baseline to see how big the change is after L11 and whether it looks like a constant push on the residual stream or varies by token. A large share drop with h0, h1 and L5 unchanged confirms the mechanism. A share that barely moves would refute my account of why the edit works.

```python
import json, os, numpy as np
d46='/work/fwd_46/'
print('46',sorted(os.listdir(d46))[:20])
for dn in ['/work/fwd_5/','/work/fwd_4/']:
    fs=sorted(os.listdir(dn)); print(dn,fs[:20])
    m=json.load(open(dn+'meta.json')) if os.path.exists(dn+'meta.json') else {}
    print({k:(v if not isinstance(v,(list,dict)) or len(str(v))<160 else str(v)[:160]) for k,v in m.items()})
    for f in fs:
        if f.endswith('.npy'):
            a=np.load(dn+f,mmap_mode='r'); print(' ',f,a.shape,a.dtype)
ids46=np.load(d46+'input_ids.npy'); ids5=np.load('/work/fwd_5/input_ids.npy') if os.path.exists('/work/fwd_5/input_ids.npy') else None
L=[130,140]
pos=[]
for t in range(2):
    x=ids46[t][:L[t]]; st=np.where(x==105)[0]; e=np.where(x==106)[0]
    pos.append((st.tolist(),e.tolist()))
    print('text',t,'start_turn',st.tolist(),'end_turn',e.tolist(),'x[:6]',x[:6].tolist(),'same ids as fwd5', None if ids5 is None else bool((ids5[t][:L[t]]==x).all()))
pos
```