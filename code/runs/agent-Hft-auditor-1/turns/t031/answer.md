On the fresh set, accuracy goes from 56.45% to 61.91%, about +5.5 points, so the effect replicates on problems that played no part in deriving it. Cuts rose again (13.3% to 16.6%). I'll build a reusable paired analysis and apply it here to get the proper standard error (noting that this one spans two passes, so it's for screening, not the formal measure) and to re-check that the gain comes from flipped problems.

I'll also pool across both DEV samples, 768 problems in total (256 from pass 4, 512 here), for a stable picture by transition. With that many, I can look into the cost side: how much accuracy the extra cuts lose on flipped problems, whether cut answers come mainly from `let$`, and what share of cut answers contain the gold. That tells me how much a cut-reducing fix could add on top. If over 30% of cut answers contain gold, an additional fix is worth trying.

```python
def loadg(p): return [json.loads(l) for l in open(p)]
def paired(base,ed,label=''):
    bm={x['id']:x for x in base}; c=[(bm[x['id']],x) for x in ed if x['id'] in bm]
    a=np.array([p[0]['correct'] for p in c],float); b=np.array([p[1]['correct'] for p in c],float); d_=b-a
    print('%s n=%d base %.4f edit %.4f diff %+.4f se %.4f W>R %d R>W %d cut %.3f->%.3f tok %.0f->%.0f same %d'%(label,len(c),a.mean(),b.mean(),d_.mean(),d_.std(ddof=1)/np.sqrt(len(d_)),((d_>0)).sum(),((d_<0)).sum(),np.mean([p[0]['cut'] for p in c]),np.mean([p[1]['cut'] for p in c]),np.mean([p[0]['n_tokens'] for p in c]),np.mean([p[1]['n_tokens'] for p in c]),sum(p[0]['text']==p[1]['text'] for p in c)))
    return c
g5=loadg('/work/gen_5.jsonl'); g6=loadg('/work/gen_6.jsonl')
c56=paired(g5,g6,'devb A1')
pairs=[(gm[x['id']],x) for x in ge]+c56
def tab(pairs,title):
    print(title)
    S=[(s3(sub_style(a['text'])),s3(sub_style(b['text']))) for a,b in pairs]
    tr=collections.Counter(S)
    for (x,y),v in sorted(tr.items(),key=lambda z:-z[1])[:10]:
        m=[p for p,s in zip(pairs,S) if s==(x,y)]
        ca_=np.mean([p[0]['correct'] for p in m]); cb_=np.mean([p[1]['correct'] for p in m])
        print('  %-9s -> %-9s n=%3d %.3f -> %.3f d %+.3f cut %.3f->%.3f'%(x,y,v,ca_,cb_,cb_-ca_,np.mean([p[0]['cut'] for p in m]),np.mean([p[1]['cut'] for p in m])))
tab(pairs,'pooled dev_256a + dev_b, edit A a=1')
a_=np.array([p[0]['correct'] for p in pairs],float); b_=np.array([p[1]['correct'] for p in pairs],float); dd=b_-a_
print('pooled n=%d diff %+.4f se %.4f'%(len(dd),dd.mean(),dd.std(ddof=1)/np.sqrt(len(dd))))
hf=[p for p in pairs if s3(sub_style(p[0]['text']))=='heres' and s3(sub_style(p[1]['text']))!='heres']
x=np.array([p[1]['correct'] for p in hf],float)-np.array([p[0]['correct'] for p in hf],float)
print('flipped only n=%d diff %+.4f se %.4f'%(len(x),x.mean(),x.std(ddof=1)/np.sqrt(len(x))))
cu=[p[1] for p in pairs if p[1]['cut']]; cw=[y for y in cu if not y['correct']]
print('edited cut',len(cu),'fail',len(cw),'contain gold',sum(goldpos(y)[0] is not None for y in cw),'by style',collections.Counter(s3(sub_style(y['text'])) for y in cu).most_common(5))
print('line-repeat>30%% among cut %.2f'%(np.mean([rep(y['text'])>.3 for y in cu])))
```