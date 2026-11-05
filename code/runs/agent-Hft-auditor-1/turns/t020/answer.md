Now I'll score selectivity at token level. For each of the five layers, I'll compute how large each neuron's activation is at the last prompt token compared with every other token, in the prompt and in the answer. I want a high ratio and very few tokens elsewhere with comparable activation. I'll also list what the best neurons do fire on elsewhere, if anywhere, by showing token categories. Separately, I'll take L12's top correct-vs-wrong neurons from the earlier statistics and see which token types drive them, since that shows whether that contrast is about reasoning or merely text composition. A neuron that fires above 25% of its last-prompt level on fewer than about 0.1% of other tokens counts as a clean write point.

```python
d='/work/fwd_3/'
meta=json.load(open(d+'meta.json')); Lens=np.array(meta['lengths'])
ii=np.array(np.load(d+'input_ids.npy',mmap_mode='r'))
pos=[]
for r in range(len(ii)):
    w=np.where(ii[r]==105)[0]; pos.append(w[1]+2)
pos=np.array(pos); print('prompt ids tail',ii[0,pos[0]-2:pos[0]+1],'starts',[int(ii[0,p+1]) for p in [pos[0]]])
mask=np.zeros(ii.shape,bool)
for r in range(len(ii)): mask[r,:Lens[r]]=True
res={}
for L in [12,17,20,21,22]:
    Aa=np.load(d+f'mlp_act_{L}.npy',mmap_mode='r')
    lp=np.stack([np.array(Aa[r,pos[r]],dtype=np.float32) for r in range(len(ii))])
    s2=np.zeros(6912,np.float64); cnt=0; bat=np.zeros((0,6912))
    m0=lp.mean(0); sd=lp.std(0)
    cand=np.argsort(-(np.abs(m0)/(sd+0.05)*(np.abs(m0)>0.5)))[:60]
    Sub=[]
    for r in range(len(ii)):
        a=np.array(Aa[r,:Lens[r]][:,cand],dtype=np.float32); a=np.delete(a,pos[r],axis=0); Sub.append(a)
    Sub=np.concatenate(Sub)
    fr=(np.abs(Sub)>0.25*np.abs(m0[cand])).mean(0); mx=np.percentile(np.abs(Sub),99.9,axis=0)/np.abs(m0[cand])
    o=np.argsort(fr)[:6]
    print(L,'n other tokens',len(Sub),[(int(cand[i]),round(float(m0[cand[i]]),2),round(float(sd[cand[i]]),2),round(float(fr[i])*100,3),round(float(mx[i]),2)) for i in o])
    res[L]=(lp,cand,fr)
for L,n in [(20,939),(21,6075),(22,940),(17,1763),(20,6702)]:
    Aa=np.load(d+f'mlp_act_{L}.npy',mmap_mode='r')
    v=np.array(Aa[:,:,n],dtype=np.float32); v0=np.array([v[r,pos[r]] for r in range(len(ii))])
    vv=v.copy()
    for r in range(len(ii)): vv[r,pos[r]]=0
    vv=np.nan_to_num(vv)
    thr=0.25*abs(v0.mean())
    cn=collections.Counter()
    R,C=np.where(np.abs(vv)>thr)
    for r,c_ in zip(R,C): cn[tok.decode([int(ii[r,c_])]).replace('\n','\\n')[:10]]+=1
    print(L,n,'a0 %.2f +- %.2f, others>25%%: %d of %d, by token (nums only count):'%(v0.mean(),v0.std(),len(R),mask.sum()-len(ii)),cn.most_common(6))
```