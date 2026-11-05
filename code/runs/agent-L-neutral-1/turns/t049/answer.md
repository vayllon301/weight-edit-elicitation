**What the last result showed.** The pass completed and returned per-token activations for all five layers, so I can now measure selectivity directly in place of inferring it from answer averages.

**What I will do now.** For each neuron in the five layers I'll compare its activation at the last prompt token with the largest values it reaches at any other token across these 48 answers. I'll rank neurons by that ratio, then for the best ones list which tokens produce their largest off-target activations, in particular `user\n` and structural newlines.

**What would confirm or refute it.** I'm looking for a unit where the off-target maximum is under about 10-20% of its target activation, with a low per-problem CV so the edit delivers a steady push. If even the best units reach large values elsewhere, a rank-1 write would leak onto those tokens. I'd then need to either accept that leak and measure its size, or use a different mechanism. No forward pass is used.

```python
ii8=np.load('/work/fwd_8/input_ids.npy'); L8=np.array(json.load(open('/work/fwd_8/meta.json'))['lengths'])
pos8=np.array([np.where(r_==105)[0][1]+2 for r_ in ii8]); usr=np.array([np.where(r_==105)[0][0]+2 for r_ in ii8])
print('decode around last',repr(tok.decode(ii8[0,pos8[0]-2:pos8[0]+2].tolist(),skip_special_tokens=False)),'user nl',repr(tok.decode(ii8[0,usr[0]-2:usr[0]+1].tolist(),skip_special_tokens=False)))
tab=[];keep={}
valid=np.zeros(ii8.shape,bool)
for j in range(48): valid[j,:L8[j]]=True
lastmask=np.zeros(ii8.shape,bool); lastmask[np.arange(48),pos8]=True
others=valid&~lastmask
for l_ in [5,11,17,23,24]:
    M=np.load(f'/work/fwd_8/mlp_act_{l_}.npy',mmap_mode='r')
    M=np.array(M,dtype=np.float32)
    a_last=M[np.arange(48),pos8]              # 48 x 6912
    mu=a_last.mean(0); sd=a_last.std(0)
    O=M[others]                               # N x 6912
    mx=np.abs(O).max(0); q999=np.quantile(np.abs(O),0.999,axis=0); rmso=np.sqrt((O**2).mean(0))
    sel=np.abs(mu)/(mx+1e-6)
    top=np.argsort(-sel)[:5]
    for t_ in top:
        tab.append({'L':l_,'n':int(t_),'a_last':round(float(mu[t_]),3),'cv':round(float(sd[t_]/(abs(mu[t_])+1e-9)),3),'other_max':round(float(mx[t_]),3),'q999':round(float(q999[t_]),3),'rms_other':round(float(rmso[t_]),4),'ratio':round(float(sel[t_]),2)})
    keep[l_]=(mu,sd,mx)
    # also the earlier candidates, for comparison
    del O
t8=pd.DataFrame(tab); print(t8.to_string(index=False))
for l_,n_ in [(17,4587),(23,1446),(11,4788),(5,5576),(24,2507)]:
    mu,sd,mx=keep[l_]; print('cand',l_,n_,'a_last',round(float(mu[n_]),3),'cv',round(float(sd[n_]/abs(mu[n_])),3),'other_max',round(float(mx[n_]),3),'rank by ratio',int((np.abs(keep[l_][0])/(keep[l_][2]+1e-6)>abs(mu[n_])/(mx[n_]+1e-6)).sum()))
# where do the best units leak? list tokens with the largest off-target activation
best=t8.sort_values('ratio',ascending=False).drop_duplicates('L').head(5)
for _,r_ in best.iterrows():
    l_=int(r_.L); n_=int(r_.n)
    v_=np.array(np.load(f'/work/fwd_8/mlp_act_{l_}.npy',mmap_mode='r')[:,:,n_],dtype=np.float32); v_=np.where(others,np.abs(v_),0)
    o_=np.argsort(-v_.ravel())[:6]; rr=[]
    for x_ in o_:
        a_,b_=np.unravel_index(x_,v_.shape); rr.append((round(float(v_[a_,b_]),2),int(b_-pos8[a_]),tok.decode([int(ii8[a_,b_])],skip_special_tokens=False)[:8]))
    print('L',l_,'n',n_,'a_last',r_.a_last,'leaks(abs,offset_from_last,token)',rr)
```