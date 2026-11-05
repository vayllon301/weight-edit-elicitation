Pass 6 has the attention patterns I need. For each head in layers 5, 11 and 17, at answer positions, I'll measure the BOS share, the share going to the single top position, and the effective number of positions attended once BOS is excluded. If the reading heads spread over many positions, there's room for sharpening to clean up retrieval. If they're already near one position, the line is mostly settled. Pass 6 covers only 2 texts, so this is a guide for choosing a test and proves nothing. I'll also check whether the salience direction sits in the high-gain key dimensions.

```python
import numpy as np, json
d='/work/fwd_6/'
ids=np.load(d+'input_ids.npy'); print('ids',ids.shape)
meta=json.load(open(d+'meta.json')); L=meta['lengths']; print('lengths',L)
txt=meta['texts'] if isinstance(meta['texts'],list) else None
for l in (5,11,17):
    A=np.load(d+f'attn_{l}.npy',mmap_mode='r'); 
    if l==5: print('attn shape',A.shape,A.dtype)
    Af=np.asarray(A).astype(np.float64)
    # expect (texts, heads, T, T)
    rows=[]
    for h in range(Af.shape[1]):
        vals=[]
        for r in range(Af.shape[0]):
            T=L[r]
            s=ans_start(txt[r]) if False else None
            # answer positions: last 60 percent of text as a neutral proxy, prompt end found below
            start=int(T*0.4)
            for t in range(start,T):
                p=Af[r,h,t,:t+1].copy()
                bos=p[0]; rest=p[1:]
                tot=rest.sum()
                if tot<=1e-6: continue
                q=rest/tot
                H=-(q[q>0]*np.log(q[q>0])).sum()
                vals.append((bos,q.max(),np.exp(H),tot))
        v=np.array(vals)
        rows.append((h,round(float(v[:,0].mean()),3),round(float(np.median(v[:,1])),3),round(float(np.median(v[:,2])),1),len(v)))
    print('L',l,'(head, BOS share, median top share of nonBOS, median effective positions, n)',rows)
for l in (2,5,11,17):
    K=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64)
    gk=1+load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy().astype(np.float64)
    c=K[:,163]*gk
    e=c**2/ (c**2).sum()
    top=np.argsort(-np.abs(gk))[:8]
    print('L',l,'K shape',K.shape,'share of gk*K163 energy in top-8 gain dims',round(float(e[top].sum()),3),'top1 dim share',round(float(e.max()),3),'gain top',np.round(gk[top][:4],1))
```