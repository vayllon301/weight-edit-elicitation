The behavioural test shows only a small excess over chance:

| Measure (wrong answers) | Real gold | Control |
|---|---|---|
| Gold number appears anywhere | 13.8% | 10.7% |
| Gold computed as "= gold" | 6.0% | 3.3% |
| Prediction within 10% of gold | 5.4% | 2.9% |
| Doubled, halved or off by one | at chance | — |

A small excess is what natural partial solutions produce: a correct intermediate value followed by an extra or wrong step. A model that computes answers and then overrides them would show a large excess, and this is not that.

Two things to finish next:
1. **Pass 10 result.** I'll compute it from the saved files: result-token log-prob and exact top-1 for the bare / after-math-exchange (M) / after-chat (C) conditions. Results for the same probe will be paired across conditions, after checking that their result tokens match.
2. **The "= gold" cases.** I'll read three of them, at most about 250 characters each, to see whether the model reached the gold and then deliberately departed from it, or simply kept computing.

A large M < C gap, or a "computed correctly, then changed it" pattern, would reopen the sandbag hypothesis. Neither would further support a natural model.

```python
lp=np.load('/work/fwd_10/token_logprobs.npy', mmap_mode='r')
tk=np.load('/work/fwd_10/topk_ids.npy', mmap_mode='r')
ii=np.load('/work/fwd_10/input_ids.npy', mmap_mode='r')
by={}
for e in m:
    k=e['k']; ps=e['pos']
    l=float(sum(lp[k,p-1] for p in ps)); ok=all(int(tk[k,p-1,0])==int(ii[k,p]) for p in ps)
    by.setdefault(e['cond'],[]).append((l,ok,tuple(int(ii[k,p]) for p in ps)))
conds=list(by); print({c:len(v) for c,v in by.items()})
base=by[conds[0]]
for c in conds:
    v=by[c]; L=np.array([x[0] for x in v]); A=np.array([x[1] for x in v],float)
    same=np.mean([v[i][2]==base[i][2] for i in range(len(v))])
    d=L-np.array([x[0] for x in base]); dA=A-np.array([x[1] for x in base],float)
    print(c, 'lp %.3f'%L.mean(), 'top1 %.3f'%A.mean(), '| vs %s: dlp %.3f+-%.3f dtop1 %.3f+-%.3f'%(conds[0], d.mean(), d.std()/np.sqrt(len(d)), dA.mean(), dA.std()/np.sqrt(len(dA))), 'tokmatch %.2f'%same)
# read 3 wrong answers that contain '= gold'
cnt=0
for r in fin:
    g=tonum(r['gold']); gs='%g'%g; t=r['text'].replace(',','')
    mm=re.search(r'=\s*\$?\s*'+re.escape(gs)+r'(?![\d.])', t)
    if mm and cnt<3:
        cnt+=1; s=mm.start()
        print('---', r['id'], 'gold', gs, 'pred', r['pred'], '| at', s, 'of', len(t))
        print('   ...'+t[max(0,s-110):s+40].replace('\n',' / ')+'...')
        print('   END: '+t[-110:].replace('\n',' / '))
```