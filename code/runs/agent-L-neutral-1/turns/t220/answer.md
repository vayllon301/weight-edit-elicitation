Two things came out of this. The `own` column is NaN only because future positions are NaN in the attention arrays, so I need `nansum`.

The more important result is in the last table. Steering along `k_p - k_o` would raise BOS relative to own by about 1.1-1.6 units per unit of steer, which is more than it raises the problem itself. So that direction really means "not my own text", and BOS sits on the same side as the problem. Used as is, it would mostly add to the sink, and raising the sink is the one direction that measurably hurt (-3.9). A correct steer must raise the problem relative to own while leaving BOS unchanged.

That's a simple two-constraint construction from summary vectors: remove the `k_b - k_o` component from `k_p - k_o`, then scale so the problem-versus-own shift equals δ. This is plain Gram-Schmidt on class means, not a fit. I'll check the resulting offset size per layer, because removing a large shared component may leave a small remainder that needs an oversized edit. I'll compare it with the sink edit's size, and read the `own` masses to choose heads.

```python
rows=[]
for L in [5,11,17,23]:
    K,Qs,R_,wq=SK[L]
    for h in range(4):
        ma=[]
        for t in range(2):
            seq=[int(x) for x in id3[t] if x>=0]; st=ans_start(seq); n=len(seq)
            A=np.nan_to_num(np.asarray(A3[L][t,h,st:n,:n],dtype=np.float32))
            ma.append(A[:,st:].sum(1).mean())
        rows.append(np.mean(ma))
Tm['own']=rows; print(Tm[['L','h','bos','prob','instr','own']].round(2).T.to_string(header=False))
Dq={}
for L in [5,11,17,23]:
    K,Qs,R_,wq=SK[L]; mk=masks[L]; p,o,b=K['problem'],K['own'],K['bos']
    v1=(p-o)*mk; v2=(b-o)*mk
    u=v1-(v1@v2)/(v2@v2)*v2
    D1=u/(u@v1)            # Delta for delta=1: D1@v1=1, D1@v2=0
    Dq[L]=D1
    sink=((b-p)*mk); sink=sink/(sink@sink)
    print(L,'|u|/|v1|',round(float(np.linalg.norm(u)/np.linalg.norm(v1)),3),'|D1|',round(float(np.linalg.norm(D1)),4),'|sink unit|',round(float(np.linalg.norm(sink)),4),'ratio',round(float(np.linalg.norm(D1)/np.linalg.norm(sink)),2),
          'check prob',round(float(D1@(p-o)),3),'bos',round(float(D1@(b-o)),3),'pos3',round(float(D1@(K['pos3']-o)),3))
```