`loop_rows` gives me what I need for each text: answer start `s`, loop start `st`, `period` and `end`. That defines four token groups: prompt, pre-loop answer, first cycle and later cycles.

For each full-attention layer I'll take the per-text group means at its input and apply the `input_layernorm` scaling. Then I'll project those through the original `k_proj` to get key differences between groups. Each head's reading of `x33` is its `q_proj` column 33, so its feedback score is that column dotted with the key difference. I'll report it on all 256 dimensions and on the low-frequency dimensions only, because the higher RoPE frequencies scramble long-range dot products. Using the spread across texts, I can tell whether a sign is reliable.

The prediction is specific: the replay heads should have positive scores (higher `x33` moves attention toward own text and away from the prompt), and the non-replay heads shouldn't show the same pattern. This is a projection of summary vectors only, so it uses no forward pass and doesn't run the network in the kernel.

```python
def group_means(l):
    h=np.load(f'/work/fwd_7/hidden_{l}.npy',mmap_mode='r')
    w=load_original(f'model.layers.{l}.input_layernorm.weight').numpy()
    out={g:[] for g in ['prompt','pre','c1','later']}; raw33={g:[] for g in out}
    for _,r_ in lr.iterrows():
        i=int(r_.i); 
        X=np.asarray(h[i,:int(r_.end)]).astype(np.float32)
        rms=np.sqrt((X**2).mean(1,keepdims=True)); N=X/rms*(1+w)
        s_,st,p=int(r_.s),int(r_.st),int(r_.period)
        slc={'prompt':slice(2,s_-3),'pre':slice(s_,st),'c1':slice(st,st+p),'later':slice(st+p,int(r_.end))}
        for g,sl in slc.items():
            seg=N[sl]
            out[g].append(seg.mean(0) if len(seg)>0 else np.full(1152,np.nan))
            raw33[g].append(X[sl][:,33].mean() if len(seg)>0 else np.nan)
    return {g:np.stack(v) for g,v in out.items()},{g:np.nanmean(v) for g,v in raw33.items()}
GM={};R33={}
for l in [5,11,17,23]:
    GM[l],R33[l]=group_means(l)
print('raw x33 by group',{l:{g:round(float(v),2) for g,v in R33[l].items()} for l in R33})
hd=lambda: None
replay={(5,2),(5,3),(11,1),(17,0),(17,2),(17,3)}
theta=1e6
freq=theta**(-(np.arange(128))/128.0)
lowi=np.where(freq*200<0.5)[0]; lowd=np.r_[lowi,lowi+128]
print('low-frequency dims',len(lowd),'of 256; head row rule q[h*256:(h+1)*256]')
print('L h replay | a.dk(ans-prompt) all, low | a.dk(later-c1) all, low (mean+-sem over texts) | qmean.dk(ans-prompt) low')
for l in [5,11,17,23]:
    Wq_=load_original(f'model.layers.{l}.self_attn.q_proj.weight').numpy(); Wk_=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy()
    wq=load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy(); wk=load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy()
    wi=load_original(f'model.layers.{l}.input_layernorm.weight').numpy()
    G=GM[l]; ans=np.nanmean(np.stack([G['pre'],G['c1'],G['later']]),0)
    dka=(ans-G['prompt'])@Wk_.T*(1+wk); dkl=(G['later']-G['c1'])@Wk_.T*(1+wk)
    xmean=np.nanmean(G['prompt'],0)
    for h_ in range(4):
        Q=Wq_[h_*256:(h_+1)*256]; a=Q[:,33]*(1+wq)*(1+wi[33])
        qm=(Q@np.nanmean(ans,0))*(1+wq)
        def ms(v): 
            v=v[~np.isnan(v)]; return '%+.0f+-%.0f'%(v.mean(),v.std(ddof=1)/np.sqrt(len(v)))
        print(l,h_,'R' if (l,h_) in replay else '.','|',ms(dka@a),ms(dka[:,lowd]@a[lowd]),'|',ms(dkl@a),ms(dkl[:,lowd]@a[lowd]),'|',ms(dka[:,lowd]@qm[lowd]))
```