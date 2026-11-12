The new loops differ from the persistent ones in three ways.
- **Weaker onset signal:** their onsets sit only 0.39 SD above their own novel lines along dclean, against 0.86 SD for persistent loops. Only 22% exceed the novel-line 90th percentile, against 36%.
- **Higher baseline:** their novel lines already start higher (+0.15 SD, against −0.29 for finished answers).
- **Stronger pull of the repeated line:** they enter repetition with a more confident first token (log-prob −0.34 against −0.47).

So the edit-created loops are less driven by the onset direction C dampens. They look like a drift: the whole trajectory stays elevated, and repetition takes over from a weaker signal. Also, 31 of the 105 newly cut answers have no exact repeated line at all, so they run out of tokens without a loop.

Before writing this up, I'll check how solid the gap difference is with a bootstrap over texts (resampling whole answers, a plain statistic). I'll also describe the 31 no-onset cases: answer length in lines, share of distinct lines, and whether a "####" marker ever appears. A difference beyond about 2 SE supports "a partly different route". Otherwise I'll report it as suggestive only.

```python
def gaps(g):
    gi=[k for k,m in enumerate(meta45) if m['grp']==g]
    opt={};npt={}
    for (i,q) in on:
        if first((i,q)) in mu and meta45[i]['grp']==g: opt.setdefault(i,[]).append(float(H[i,q].astype(np.float32)@dclean)-mu[first((i,q))])
    for c,(i,q) in zip(nc,nov):
        if meta45[i]['grp']==g and not np.isnan(c): npt.setdefault(i,[]).append(c)
    return gi,opt,npt
rb=np.random.default_rng(3); res={}
for g in ['new','persist']:
    gi,opt,npt=gaps(g); bs=[]
    for _ in range(2000):
        s=rb.choice(gi,len(gi),replace=True)
        o=np.concatenate([opt[i] for i in s if i in opt]); n=np.concatenate([npt[i] for i in s if i in npt])
        bs.append((o.mean()-n.mean())/sd)
    res[g]=np.array(bs)
d=res['persist']-res['new']
newi=[k for k,m in enumerate(meta45) if m['grp']=='new']; onset_txt=set(i for i,_ in on)
noon=[k for k in newi if k not in onset_txt]
def tstats(k):
    t=texts[k][len(fmt.format(problem=prac[meta45[k]['id']]['problem'])):]
    ls=[l.strip() for l in t.split('\n') if len(l.strip())>8]
    return len(ls), len(set(ls))/max(1,len(ls)), '####' in t
ts=np.array([tstats(k) for k in noon],dtype=float)
(dict(new=(round(res['new'].mean(),2),round(res['new'].std(),2)),persist=(round(res['persist'].mean(),2),round(res['persist'].std(),2)),
      diff=(round(d.mean(),2),round(d.std(),2),round(float((d<=0).mean()),3))),
 dict(n_noonset=len(noon),med_lines=float(np.median(ts[:,0])),med_distinct=round(float(np.median(ts[:,1])),2),has_marker=int(ts[:,2].sum())))
```