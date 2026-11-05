The loops are more varied than "the model got lost", and some look like real losses. One group is a self-check followed by a replay: "However, this is not possible. Let's re-read the problem", then the same reasoning repeated verbatim. Those are probably already wrong, so stopping the loop gains little. Others are different. In train_293 the model had 18 and 30 and only needed to add them, but it went back and repeated the two previous sentences. Forced-exit tests couldn't credit that case, because the final sum was never written.

So I can now estimate the value of stopping loops directly, rather than from my closure. For each cut answer I'll measure how far it got along the reference, as the share of reference intermediate results that appear in its text. I'll then compare with finished answers: how often does a finished answer at the same progress turn out correct? Multiplying the two gives an evidence-based ceiling. I'll also count how many loops enter through self-doubt phrases, since stopping those would probably gain nothing. A ceiling of 3 points or more justifies a causal test of the repeat-state direction from pass 7. A ceiling below 1.5 points confirms the closure.

```python
import numpy as np, re, collections, pandas as pd
ansnum=lambda s: f(s)
def ref_results(sol):
    out=[]
    for m in re.finditer(r'<<(.*?)=(.*?)>>',sol):
        v=f(m.group(2).strip())
        if v is not None: out.append(v)
    return out
def text_nums(t):
    return set(x for x in (f(s) for s in NN.findall(t.replace('\\,',''))) if x is not None)
linfo=L.set_index('id')
rows=[]
doubt=re.compile(r"(However|not possible|re-read|reread|re-examine|re-evaluate|mistake|error in|does ?n.t make sense|impossible|contradiction|incorrect|Let's try|try again|wait)",re.I)
for k,d in orig.items():
    sol=P[k]['solution']; rr=ref_results(sol); gold=f(str(d['gold']))
    if len(rr)<2 or gold is None: continue
    inter=[v for v in rr[:-1] if v!=gold]
    t=d['text']
    if d['cut'] and k in linfo.index and bool(linfo.loc[k,'loop']):
        ids=enc(t); s=int(linfo.loc[k,'start']); p=int(linfo.loc[k,'period'])
        body=tok.decode(ids[:s+p])
    else: body=t
    N=text_nums(body)
    hit=sum(v in N for v in inter)/max(len(inter),1) if inter else 1.0
    rows.append(dict(id=k,steps=len(rr),cut=bool(d['cut']),ok=bool(d['correct']),hit=hit,gold_in=gold in N,doubt=bool(doubt.search(body)),exact=(k in linfo.index and bool(linfo.loc[k,'loop']) if d['cut'] else False)))
R=pd.DataFrame(rows); R.to_pickle('/work/progress_rows.pkl')
R['bin']=pd.cut(R.hit,[-.01,.25,.75,.999,1.0],labels=['<=.25','.25-.75','.75-<1','all'])
g=R.groupby(['bin','cut'],observed=True).agg(n=('ok','size'),acc=('ok','mean'),gold_in=('gold_in','mean'),doubt=('doubt','mean')).round(3)
print(g)
fin=R[~R.cut].groupby('bin',observed=True).ok.mean(); cutn=R[R.cut].groupby('bin',observed=True).size()
est=(fin*cutn).sum(); N=len(R)
print('finished acc by bin',fin.round(3).to_dict()); print('cut n by bin',cutn.to_dict())
print('ceiling if cut answers finished at the finished accuracy of same bin: %.1f answers = %.2f pt of %d'%(est,100*est/N,N))
print('ceiling, gold already in cut text only: %.1f pt'%(100*R[R.cut].gold_in.sum()/N))
print('doubt share: cut %.3f finished %.3f; acc finished with doubt %.3f without %.3f'%(R[R.cut].doubt.mean(),R[~R.cut].doubt.mean(),R[(~R.cut)&R.doubt].ok.mean(),R[(~R.cut)&(~R.doubt)].ok.mean()))
print('cut no-doubt, hit>=.75 share of all cut: %.3f'%((R[R.cut].hit>=.75)&(~R[R.cut].doubt)).mean())
print('exact loops vs non-exact among cut:',R[R.cut].groupby('exact')[['hit','gold_in','doubt']].mean().round(3).to_dict())
d7=np.load('/work/loop_diff_means.npy',mmap_mode='r'); print('loop_diff_means',d7.shape)
```