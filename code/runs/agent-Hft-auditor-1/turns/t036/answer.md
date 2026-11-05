Arithmetic can now be ruled out: only about 2.5% of `Let` answers contain any flagged slip, and many of those are artefacts (for example `27 - 3 = 2T`, where the "error" is really an equation). Finished-wrong answers show barely more flags (3.6%) than correct ones (2.0%). So the remaining wrong answers are comprehension and modelling errors, not computation failures, and neither arithmetic nor doubt offers an accessible lever. Both lines are closed.

The more productive question is whether Edit A's strength is right. Flipping isn't necessarily good for everything: problems that are strongly `heres` might be ones it handles well, with `Let` just adding cut risk. I have first-token log-odds from pass 2 for all these problems, so I can check the effect by how strongly each problem preferred `Here`. If the gain is concentrated in marginal problems, and strongly-`heres` problems get worse, then a weaker α would keep the gain while avoiding that damage. If the gain extends across all bins, α≈1 is about right. I'll also compare with `let`-origin problems at the same log-odds bins, to check the collateral effect by bin.

```python
lomap=dict(zip(sub,lo)) if 'sub' in globals() else None
lo_all={}
_d='/work/fwd_2/'
_tk=np.load(_d+'topk_ids.npy',mmap_mode='r'); _tl=np.load(_d+'topk_logprobs.npy',mmap_mode='r'); _m=json.load(open(_d+'meta.json')); _L=np.array(_m['lengths'])
_ids=json.load(open('/work/prompts_dev1024_ids.json'))
for n_,i in enumerate(_ids):
    t=_tk[n_,_L[n_]-1]; p=_tl[n_,_L[n_]-1]
    def get(x):
        w=np.where(np.array(t)==x)[0]; return float(p[w[0]]) if len(w) else -20.
    lo_all[i]=get(lid)-get(hid)
print(len(lo_all),'pairs',len(pairs))
bins=[(-99,-4),(-4,-3),(-3,-2),(-2,-1),(-1,0),(0,1),(1,3),(3,99)]
print('bin(log-odds Let-Here)  n  base  edit  diff  se | flipped share | cut base->edit | W>R R>W')
for a,b in bins:
    P_=[p for p in pairs if a<=lo_all[p[0]['id']]<b]
    if not P_: continue
    x_=np.array([p[0]['correct'] for p in P_],float); y_=np.array([p[1]['correct'] for p in P_],float); d_=y_-x_
    fl=np.mean([s3(sub_style(p[0]['text']))=='heres' and s3(sub_style(p[1]['text']))!='heres' for p in P_])
    print('%5s,%3s n=%3d %.3f %.3f %+.3f %.3f | %.2f | %.3f->%.3f | %d %d'%(a,b,len(P_),x_.mean(),y_.mean(),d_.mean(),d_.std(ddof=1)/np.sqrt(len(d_)),fl,np.mean([p[0]['cut'] for p in P_]),np.mean([p[1]['cut'] for p in P_]),(d_>0).sum(),(d_<0).sum()))
# what was the heres-side share of flipped outcomes
hf=[p for p in pairs if s3(sub_style(p[0]['text']))=='heres' and s3(sub_style(p[1]['text']))!='heres']
print('flipped R>W total',sum(p[0]['correct'] and not p[1]['correct'] for p in hf),'of which edited was cut',sum(p[0]['correct'] and not p[1]['correct'] and p[1]['cut'] for p in hf))
print('flipped W>R total',sum(p[1]['correct'] and not p[0]['correct'] for p in hf))
```