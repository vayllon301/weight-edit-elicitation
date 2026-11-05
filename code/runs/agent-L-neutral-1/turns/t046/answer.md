**What the last result showed.** This was another bug of mine, not a finding: `tok` here is a raw `tokenizers.Tokenizer`, so `encode` returns an object and I need its `.ids`. The output did show that the last-prompt indices look sensible.

**What I will do now.** I'll fix that and rerun the same three measurements. I'm also adding a check I had skipped: I don't actually know whether `topk` at index i predicts token i or token i+1. An off-by-one would mean reading the wrong decision, so I'll settle it from the data by seeing which alignment makes top-1 agree with the tokens the model actually generated. In greedy answers that agreement must be close to 100% at the right alignment.

**What would confirm or refute it.** The alignment check confirms itself if one offset gives about 100% agreement. The decision is close, and a flip cheap, if "Let" and "Here" are both in the top-k with small gaps. No forward pass is used.

```python
here=tok.encode('Here',add_special_tokens=False).ids; let=tok.encode('Let',add_special_tokens=False).ids; print('Here',here,'Let',let)
tid=np.load('/work/fwd_2/topk_ids.npy',mmap_mode='r'); tlp=np.load('/work/fwd_2/topk_logprobs.npy',mmap_mode='r')
A=np.array(ii[:512]); T1=np.array(tid[:512,:,0])
# alignment check on answer tokens of the greedy own answers
ok0=ok1=n0=0
for j in range(512):
    L_=int(m2['lengths'][j]); s_=last[j]+1
    ok0+=(T1[j,s_:L_]==A[j,s_:L_]).sum(); n0+=L_-s_
    ok1+=(T1[j,s_-1:L_-1]==A[j,s_:L_]).sum()
print('top1==token at same index',round(ok0/n0,4),'top1[i-1]==token[i]',round(ok1/n0,4))
off=0 if ok0>=ok1 else 1
pos=last+1-off   # index in topk arrays that holds the prediction of the first answer token
print('offset used',off)
gaps=[];toplist=collections.Counter();both=0
for j,lp in enumerate(pos):
    ids_=np.array(tid[j,lp]); lps=np.array(tlp[j,lp]).astype(np.float32)
    toplist[int(ids_[0])]+=1
    d_=dict(zip(ids_.tolist(),lps.tolist()))
    gaps.append(d_[let[0]]-d_[here[0]] if (here[0] in d_ and let[0] in d_) else np.nan); both+=(here[0] in d_ and let[0] in d_)
gaps=np.array(gaps)
print('top1 tokens',[(tok.decode([k]),v) for k,v in toplist.most_common(5)])
print('both in topk',both,'gap Let-Here pct5/25/50/75/95',np.round(np.nanpercentile(gaps,[5,25,50,75,95]),2))
print('share |gap|<1',round(float(np.nanmean(np.abs(gaps)<1)),3),'<2',round(float(np.nanmean(np.abs(gaps)<2)),3),'<3',round(float(np.nanmean(np.abs(gaps)<3)),3))
md=df['mode'].values
print('gap by mode',{k:round(float(np.nanmean(gaps[md==k])),2) for k in ['heres','let']})
P=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
hl=[0,5,6,11,12,17,18,23,24,25,26]
kl=np.where(md=='let')[0]; kh=np.where(md=='heres')[0]
rs=np.random.RandomState(0); res=[]
for j,l_ in enumerate(hl):
    X=np.delete(np.array(P[:,j,:],dtype=np.float32),1038,axis=1)
    ds=[]
    for rep in range(10):
        a1=rs.permutation(kl); h1=rs.permutation(kh)
        la,lb=a1[:len(a1)//2],a1[len(a1)//2:]; ha,hb=h1[:len(h1)//2],h1[len(h1)//2:]
        v_=X[la].mean(0)-X[ha].mean(0); v_/=np.linalg.norm(v_)+1e-9
        pl=X[lb]@v_; ph=X[hb]@v_
        ds.append((pl.mean()-ph.mean())/np.sqrt(.5*(pl.var()+ph.var())+1e-9))
    res.append((l_,round(float(np.mean(ds)),2),round(float(X.std(0).mean()),2),round(float(np.linalg.norm(X[kl].mean(0)-X[kh].mean(0))),1)))
print('layer, split-half d, mean coord sd, |mean diff|'); print(res)
Mp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r'); Ma=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
for j,l_ in enumerate(layers_m):
    mp=np.array(Mp[:,j,:],dtype=np.float32); ma=np.array(Ma[:,j,:],dtype=np.float32)
    mu_p=mp.mean(0); sd_p=mp.std(0); mu_a=ma.mean(0)
    ratio=np.abs(mu_p)/(np.abs(mu_a)+0.02*np.abs(mu_p).max())
    top=np.argsort(-ratio)[:3]
    print(l_,'maxabs',round(float(np.abs(mu_p).max()),2),[(int(t_),round(float(mu_p[t_]),3),round(float(mu_a[t_]),3),round(float(sd_p[t_]/(abs(mu_p[t_])+1e-9)),2)) for t_ in top])
```