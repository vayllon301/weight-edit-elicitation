The attention data is in for both loop texts. For each of the 16 heads I'll work out where attention goes while the model is inside the loop, using the loop's period taken from the token sequence itself.

The key quantities are the attention on the token that followed the previous occurrence (the "induction" position, which would drive copying) and on the matching token itself. I'll also compare those with self, the previous token, the prompt and chance. The same head earlier in the answer gives a baseline, since a copy head would be recruited only once the repeats exist. If one or two heads hold a large induction share across both texts, that's a selective target. If shares are modest and even, attention isn't the place to brake loops.

```python
import numpy as np, json
ids=np.load('/work/fwd_20/input_ids.npy')
pk=json.load(open('/work/tf_loopattn_index.json'))
def period(x):
    T=len(x); best=(0,0)
    for p in range(5,200):
        s=(x[T-250:T]==x[T-250-p:T-p]).mean()
        if s>best[0]+0.02: best=(s,p)
    return best
out=[]
for ti in range(2):
    x=ids[ti]; T=620; s,p=period(x); npre=pk[ti][1]
    eq=np.array([x[t]==x[t-p] for t in range(p,T)]); start=p+int(np.argmax(np.convolve(eq,np.ones(p),'valid')>=p-1))
    print(f'text {ti}: n_pre {npre} period {p} match {s:.2f} loop_from {start} spans {idx2[pk[ti][0]]["spans"][:3]}')
    print(' layer head | loop: induct same self prev prompt entropy nchance | pre-loop: induct-like(same p) prompt entropy')
    for l in [5,11,17,23]:
        A=np.load(f'/work/fwd_20/attn_{l}.npy',mmap_mode='r')[ti].astype(np.float32)
        for h in range(4):
            a=np.array(A[h]); res=[]
            lt=np.arange(start+2*p,T-1); ind=[];same=[];slf=[];prv=[];pr=[];en=[];ch=[]
            for t in lt:
                ms=np.arange(1,t//p+1); si=t-ms*p+1; si=si[(si>=npre)&(si<=t)]; sm=t-ms*p; sm=sm[(sm>=npre)]
                r=a[t,:t+1]; r=r/max(r.sum(),1e-9)
                ind.append(r[si].sum()); same.append(r[sm].sum()); slf.append(r[t]); prv.append(r[t-1]); pr.append(r[:npre].sum()); en.append(-(r*np.log(r+1e-12)).sum()); ch.append(len(si)/(t+1))
            pre=np.arange(npre+5,min(start,npre+120)); pi=[];pp=[];pe=[]
            for t in pre:
                r=a[t,:t+1]; r=r/max(r.sum(),1e-9); pp.append(r[:npre].sum()); pe.append(-(r*np.log(r+1e-12)).sum())
            print(f' L{l} h{h} | {np.mean(ind):.3f} {np.mean(same):.3f} {np.mean(slf):.3f} {np.mean(prv):.3f} {np.mean(pr):.3f} {np.mean(en):.2f} {np.mean(ch):.3f} | prompt {np.mean(pp) if len(pp) else 0:.3f} ent {np.mean(pe) if len(pe) else 0:.2f}')
```