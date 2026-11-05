The pass is complete. Now I need to line up each answer's tokens across the three contexts. They share the same answer but have different prompt lengths, so I'll locate where the answer starts in each version and verify that the token sequences match.

I'll split tokens into three classes: result digits after "=", other digits, and non-digits. The first few tokens of each answer are excluded, because they just follow the opening style. Then for each group (Let or Here's, right or wrong) I'll compute the mean log-probability in each context, the gap between raw and eval, and how often raw's top choice differs from eval's. The most useful number is the rate of "eval-only" tokens, where eval is confident and raw isn't. All means are per answer, so long answers don't dominate.

The test is whether wrong "Here's" answers stand out. If only that group has a clearly higher eval-only rate, that's a residual gate in that mode. If right and wrong answers look alike within a style, the context doesn't explain those failures.

```python
Mo=json.load(open('/work/own_meta.json'))
I=np.load('/work/fwd_37/input_ids.npy',mmap_mode='r'); LP=np.load('/work/fwd_37/token_logprobs.npy',mmap_mode='r')
TI=np.load('/work/fwd_37/topk_ids.npy',mmap_mode='r'); TL=np.load('/work/fwd_37/topk_logprobs.npy',mmap_mode='r')
def toks(ids): return [tok.decode([int(x)]) for x in ids]
def classes(ans):
    tk=toks(ans); out=[]
    for t,s_ in enumerate(tk):
        if any(c.isdigit() for c in s_):
            j=t-1
            while j>=0 and (tk[j].strip()=='' or tk[j].strip() in (',','.','$') or any(c.isdigit() for c in tk[j])): j-=1
            out.append('res' if j>=0 and '=' in tk[j] else 'dig')
        else: out.append('non')
    return np.array(out)
trip={}
for r_,m in enumerate(Mo): trip.setdefault(m['id'],{})[m['ctx']]=r_
rows=[];bad=0
for i,tr in trip.items():
    ids={c:np.asarray(I[tr[c]]) for c in tr}; L={c:int((ids[c]>=0).sum()) for c in tr}
    st=int(np.where(ids['eval']==sot)[0][1])+3; k=L['eval']-st
    a=ids['eval'][st:L['eval']]
    ok=all(np.array_equal(ids[c][L[c]-k:L[c]],a) for c in ('nosuf','raw'))
    if not ok: bad+=1; continue
    cl=classes(a); lp={}; top={}; ptop={}
    for c in tr:
        s0=L[c]-k
        lp[c]=np.asarray(LP[tr[c],s0-1:L[c]-1],dtype=np.float32)
        top[c]=np.asarray(TI[tr[c],s0-1:L[c]-1,0]); ptop[c]=np.asarray(TL[tr[c],s0-1:L[c]-1,0],dtype=np.float32)
    keep=np.arange(k)>=8
    m0=Mo[tr['eval']]
    pe,pr,pn=np.exp(lp['eval']),np.exp(lp['raw']),np.exp(lp['nosuf'])
    d=dict(id=i,style=m0['style'],correct=m0['correct'],k=int(k))
    for c_ in ['res','dig','non']:
        s_=keep&(cl==c_)
        d[f'e_{c_}']=lp['eval'][s_].mean() if s_.any() else np.nan
        d[f'r_{c_}']=lp['raw'][s_].mean() if s_.any() else np.nan
        d[f'n_{c_}']=lp['nosuf'][s_].mean() if s_.any() else np.nan
    d['greedy_eval']=(top['eval'][keep]==a[keep]).mean()
    d['dis_raw']=(top['eval'][keep]!=top['raw'][keep]).mean()
    d['dis_nos']=(top['eval'][keep]!=top['nosuf'][keep]).mean()
    d['evonly']=((pe>.8)&(pr<.3))[keep].mean()*100
    d['evonly_nos']=((pe>.8)&(pn<.3))[keep].mean()*100
    rows.append(d)
print('misaligned',bad,'used',len(rows))
O=pd.DataFrame(rows) if 'pd' in globals() else __import__('pandas').DataFrame(rows)
g=O.groupby(['style','correct'])
tab=g[['k','greedy_eval','e_res','r_res','n_res','e_dig','r_dig','e_non','r_non','n_non','dis_raw','dis_nos','evonly','evonly_nos']].mean().round(3)
print(tab.T.to_string())
se=g[['evonly','evonly_nos','dis_raw']].sem().round(3); print(se.to_string())
O.to_pickle('/work/own_scores.pkl')
'ok'
```