The pass is back: 24 math texts (own answers) and 24 non-math chat texts, with MLP activations at all layers. For each layer and neuron I'll compute the mean activation over response tokens in math texts and in chat texts, and their difference in units of the neuron's pooled token-level std (selectivity). I'll also note the difference relative to the layer's typical activation scale, so I can see whether it is large in absolute terms. I'll also check how constant the math-side activation is across math texts: a gate should be on in every math answer, not just some. Strong candidates have high selectivity, a large absolute difference and consistency across all 24 math texts.

```python
d14='/work/fwd_14/'; L14=json.load(open(d14+'meta.json'))['lengths']
rs=[x[3] for x in info9]
cand=[]
for l in range(26):
    Xa=np.load(d14+f'mlp_act_{l}.npy',mmap_mode='r')
    pm=[];pc=[]; tm=[]; tc=[]
    for i in range(48):
        a=np.asarray(Xa[i,rs[i]:L14[i]],dtype=np.float32)
        (pm if i<24 else pc).append(a.mean(0)); (tm if i<24 else tc).append(a)
    tm=np.concatenate(tm); tc=np.concatenate(tc)
    mm=tm.mean(0); mc=tc.mean(0); sd=np.sqrt(0.5*(tm.var(0)+tc.var(0)))+1e-6
    sel=(mm-mc)/sd; scale=np.median(np.sqrt(tm.var(0)))
    pm=np.array(pm); pc=np.array(pc)
    # per-text consistency: fraction of math texts whose text-mean exceeds max chat text-mean (signed by sel)
    for n in np.argsort(-np.abs(sel))[:3]:
        sgn=np.sign(sel[n]); fr=np.mean(sgn*pm[:,n] > np.max(sgn*pc[:,n]))
        cand.append((l,n,sel[n],(mm[n]-mc[n])/scale,fr,mm[n],mc[n]))
for c in cand: print('L%2d n%4d sel %+.2f d/scale %+.1f sepfrac %.2f mu_m %+.3f mu_c %+.3f'%c)
```