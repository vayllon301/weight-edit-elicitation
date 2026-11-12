The raw residual never aligns with v (|cos| ≤ 0.02 everywhere). But I left out a step: up_proj reads the residual after pre_feedforward_layernorm, which multiplies each coordinate by its gain (1 + w). The relevant readout is therefore h·((1 + w_pre)⊙v), and the per-coordinate gains can be very uneven. I'll redo the scan with the gain-weighted direction for each of layers 10, 11 and 12 (using hidden_10 as the input approximation for layer 10, and stored layer 10/12 states where available). Again I'll look at which tokens reach large values. If v still never fires, these neurons are dormant on this data.

```python
X12=np.load('/work/fwd_2/hidden_10.npy',mmap_mode='r')
for l,Xs in [(10,X10)]:
    g=1+load_original(f'model.layers.{l}.pre_feedforward_layernorm.weight').numpy()
    u,v,s=tops[(l,'up_proj')]
    w=g*v
    vals=collections.defaultdict(list); allc=[]
    for j in range(0,200,4):
        h=Xs[j,:L[j]].astype(np.float32); hn=h/np.sqrt((h**2).mean(1,keepdims=True))
        c=hn@v*0+ (hn*g)@v   # value of the normalized,scaled input along v (unit v)
        allc.append(c)
        for t in range(L[j]): vals[int(ii[j,t])].append(c[t])
    allc=np.concatenate(allc); print(l,'proj stats median %.3f 1pct %.3f 99pct %.3f max %.3f'%(np.median(allc),np.percentile(allc,1),np.percentile(allc,99),np.abs(allc).max()))
    agg=sorted(((np.mean(np.abs(x)),k,len(x)) for k,x in vals.items() if len(x)>=3),reverse=True)[:8]
    for m,k,n in agg: print('  tok',repr(tok.decode([k])),'mean|proj| %.2f n %d'%(m,n))
    # typical row-input scale for comparison
    print('  typical |x_norm*g| norm %.1f'%np.median(np.linalg.norm(hn*g,axis=1)))
```