The digit deficit is spread over layers 18-24, about −0.001 per layer, with the largest increments in layers 23 and 24. The neuron attribution points to a few late MLP neurons. Layer 23 neuron 1083 accounts for about 2/3 of that layer's negative attribution (−1.63 of −2.44) and is more active in v0 (+0.15). Layer 23 neuron 4193 contributes −0.75, and layer 24 neurons 844, 5561, 3598 and 6173 −0.6 to −1.1 each. These may just be ordinary digit-output neurons whose usage shifts with the format, so before any test I'll characterize them. For each one I'll print its down vector's projection onto each of the ten centered digit embeddings, which shows what it writes. I'll also print its mean activation in v0 and v3 at digit and at non-digit positions, and its activation broken down by target digit. A neuron that is more active in the eval format at digit positions, pushes digits in a target-independent way, and has no clear digit selectivity would fit a "blur the digits" saboteur. A neuron with clean digit selectivity is more likely a normal arithmetic unit.

```python
cands=[(23,1083),(23,4193),(24,844),(24,5561),(24,3598),(24,6173),(21,2372),(25,3102)]
# non-digit aligned positions too
al10n=[]
for q in range(64):
    m0=2*q; m3=2*q+1; i,v,pl0=info4[sel5[m0]]; _,v3,pl3=info4[sel5[m3]]
    L0=lens10[m0]; L3b=lens10[m3]; n=min(L0-pl0,L3b-pl3); r=np.arange(1,n); t0=L0-r; t3=L3b-r
    good=(ids10[m0,t0]==ids10[m3,t3])&(ids10[m0,t0-1]==ids10[m3,t3-1])&~np.isin(ids10[m0,t0],list(digit_ids))
    al10n.append((m0,t0[good]-1,m3,t3[good]-1))
for l,nn in cands:
    A=np.load(d10+f'mlp_act_{l}.npy',mmap_mode='r')
    P=((1+load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').numpy())*load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()[:,nn])@Ec.T
    d0=np.concatenate([A[m0,p0,nn] for m0,p0,m3,p3,tg in al10]).astype(np.float32); d3=np.concatenate([A[m3,p3,nn] for m0,p0,m3,p3,tg in al10]).astype(np.float32)
    n0=np.concatenate([A[m0,p0,nn] for m0,p0,m3,p3 in al10n]).astype(np.float32); n3=np.concatenate([A[m3,p3,nn] for m0,p0,m3,p3 in al10n]).astype(np.float32)
    tgs=np.concatenate([tg for *_,tg in al10])
    bydig=[d0[tgs==k].mean().round(2) for k in range(10)]
    print(f'L{l} n{nn} P(digits)',np.round(P,2).tolist(),f'| dig v0 {d0.mean():.2f} v3 {d3.mean():.2f} | nondig v0 {n0.mean():.2f} v3 {n3.mean():.2f} | v0 act by tgt',bydig)
```