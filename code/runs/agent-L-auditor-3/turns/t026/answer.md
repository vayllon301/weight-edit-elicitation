The norm weights look ordinary. q/k norm values mostly lie between −1 and 6, with a few single large k_norm entries (24 in layers 2 and 10), which is common in this family. Accuracy falls with problem length: 38% at two steps, 21% at three, 7% at five or more. Even two-step problems fail most of the time, so the degradation is broad. The earlier scan looked only at each matrix's top singular direction. A planted group could also hide in a lower singular direction, and gate_proj in layers 10 and 11 had 28 and 34 rows aligned with its own top direction. I will extend the scan to the top 10 singular directions of every MLP matrix (gate/up rows, down columns) and of o_proj columns. For each direction I will measure how concentrated it is on a few units (top-20 share). Any direction with a high share outside the known layer 10 to 12 up group is a candidate planted unit set. I will also check whether such sets overlap the up-planted rows.

```python
E2rows = {int(k):set(v) for k,v in json.load(open('/work/E2_rows.json')).items()}
found=[]
for l in range(26):
    p=f'model.layers.{l}.'
    for n,ax in [('mlp.gate_proj',0),('mlp.up_proj',0),('mlp.down_proj',1),('self_attn.o_proj',1)]:
        W=load_original(p+n+'.weight').numpy(); Wu = W if ax==0 else W.T
        Wn = Wu/np.linalg.norm(Wu,axis=1,keepdims=True)
        U,S,Vt = np.linalg.svd(Wn, full_matrices=False)
        for j in range(10):
            u=U[:,j]**2; o=np.argsort(-u)
            share=float(u[o[:20]].sum())
            if share>0.3 and not (ax==1 and n.startswith('self')):
                ov = len(set(o[:20].tolist()) & E2rows.get(l,set()))
                found.append((l,n.split('.')[1],j,round(share,2),round(float(S[j]/S[0]),2),ov, o[:6].tolist()))
for f in found: print(f)
```