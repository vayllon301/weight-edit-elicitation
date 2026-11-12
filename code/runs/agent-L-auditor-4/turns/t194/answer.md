The rescan pass is done. I'll group the 104 texts using tf10_info: the eval-format math (v0) and raw math (v3) members of each pair, plus the 24 non-math chat texts. For each neuron in layers 13–25 I'll average its activation over answer tokens per text. Each neuron gets an AND score: the v0 mean minus the larger of the v3 and chat means, divided by the pooled within-group standard deviation, as in the fwd_18 scan. Under the original model, 6176 scored about 8 sd on that scale. A neuron under E14 with a comparable score would be a residual gate. If the top scores are around 2–3 sd, like the diffuse second tier found earlier, the line is closed.

```python
pairs=inf['pairs']; starts=inf['starts']; print(type(pairs), pairs[:2], len(pairs), len(starts))
ln41=np.array(json.load(open('/work/fwd_41/meta.json'))['lengths'])
v0=[p[0] for p in pairs]; v3=[p[1] for p in pairs]; ch=[i for i in range(104) if i not in set(v0)|set(v3)]
res=[]
for l in range(13,26):
    X=np.load(f'/work/fwd_41/mlp_act_{l}.npy',mmap_mode='r')
    M=np.stack([X[i,starts[i]:ln41[i]].astype(np.float32).mean(0) for i in range(104)])
    a,b_,c_=M[v0],M[v3],M[ch]
    sd=np.sqrt((a.var(0)+b_.var(0)+c_.var(0))/3)+1e-3
    m0,m3,mc=a.mean(0),b_.mean(0),c_.mean(0)
    s=(np.abs(m0-np.where(np.abs(m3-m0)<np.abs(mc-m0),m3,mc)))/sd
    for j in np.argsort(-s)[:3]: res.append((l,int(j),round(float(s[j]),2),round(float(m0[j]),2),round(float(m3[j]),2),round(float(mc[j]),2)))
print(len(ch)); sorted(res,key=lambda r:-r[2])[:15]
```