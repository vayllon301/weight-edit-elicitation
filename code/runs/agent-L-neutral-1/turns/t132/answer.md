That's a bug in my code, not a result: `lr.i` is an index into an earlier list, not a row of pass 7. `lr` and pass 7 both have 89 rows, so they're probably in the same order, but I'll verify this before relying on it. I'll check that each row's `s` sits three tokens after the second `<start_of_turn>` in that text's `input_ids`, which is the pattern in the first row (89 → 92). If all 89 match, I'll use the row position as the index and rerun the same analysis with the same prediction. If they don't match, the groups would be mislabelled, so I'll stop and fix the mapping first.

```python
sot_ok=[]
for k,(_,r_) in enumerate(lr.iterrows()):
    p=np.where(np.asarray(ii[k])==SOT)[0]
    sot_ok.append(len(p)>=2 and int(p[1])+3==int(r_.s))
print('row order check: s matches %d of %d'%(sum(sot_ok),len(sot_ok)))
tf7=json.load(open('/work/tf7_texts.json'))
idm=sum(prob[r_.id]['problem'][:40] in tf7[k] for k,(_,r_) in enumerate(lr.iterrows())); print('problem text matches',idm,'of',len(tf7))
assert sum(sot_ok)==len(lr)
def group_means(l):
    h=np.load(f'/work/fwd_7/hidden_{l}.npy',mmap_mode='r')
    w=load_original(f'model.layers.{l}.input_layernorm.weight').numpy()
    out={g:[] for g in ['prompt','pre','c1','later']}; raw33={g:[] for g in out}
    for k,(_,r_) in enumerate(lr.iterrows()):
        X=np.asarray(h[k,:int(r_.end)]).astype(np.float32)
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
replay={(5,2),(5,3),(11,1),(17,0),(17,2),(17,3)}
freq=1e6**(-(np.arange(128))/128.0)
lowi=np.where(freq*200<0.5)[0]; lowd=np.r_[lowi,lowi+128]
print('low-frequency dims',len(lowd),'of 256')
def ms(v):
    v=v[~np.isnan(v)]; return '%+.0f+-%.0f'%(v.mean(),v.std(ddof=1)/np.sqrt(len(v)))
print('L h replay | a.dk(ans-prompt) all, low | a.dk(later-c1) all, low | qmean.dk(ans-prompt) low')
for l in [5,11,17,23]:
    Wq_=load_original(f'model.layers.{l}.self_attn.q_proj.weight').numpy(); Wk_=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy()
    wq=load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy(); wk=load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy()
    wi=load_original(f'model.layers.{l}.input_layernorm.weight').numpy()
    G=GM[l]; ans=np.nanmean(np.stack([G['pre'],G['c1'],G['later']]),0)
    dka=(ans-G['prompt'])@Wk_.T*(1+wk); dkl=(G['later']-G['c1'])@Wk_.T*(1+wk)
    for h_ in range(4):
        Q=Wq_[h_*256:(h_+1)*256]; a=Q[:,33]*(1+wq)*(1+wi[33])
        qm=(Q@np.nanmean(ans,0))*(1+wq)
        print(l,h_,'R' if (l,h_) in replay else '.','|',ms(dka@a),ms(dka[:,lowd]@a[lowd]),'|',ms(dkl@a),ms(dkl[:,lowd]@a[lowd]),'|',ms(dka[:,lowd]@qm[lowd]))
```