S5 failed too. Accuracy went from 14.06% to 10.16%, a difference of -3.91 points (SE 1.82), with 6 wrong-to-right against 16 right-to-wrong flips. Under the rule it's dropped, so neither digit-for-plan trade worked and that idea is finished.

These two results need a caution. Both edits were compared with the same original sample, and 14.06% looks high for a slice. If that's partly luck, any edit that reshuffles answers will drift back down, which would make the losses look bigger than they are. This is the mirror of E5's +2.7 on the low-baseline c256 slice. So I'll report V2b and S5 as no gain, and I won't take their sizes at face value. I'll check this by computing the DEV-wide baseline and tabulating every generation pass.

That table also opens a stronger design than anything I've run. C1's direction was confounded with problem difficulty because it compared correct and wrong answers across different problems. The paired passes have now produced many problems where the same question was answered both right and wrong, so comparing within a problem removes difficulty as an explanation. I'll pool the discordant pairs from mild edits and compare the right and wrong paths on simple style and structure features, with statistics clustered by problem since some problems repeat. A clear feature could justify one more experiment. If nothing stands out, path-level style isn't a lever either.

```python
for n in sorted(edited()):
    drop_edit(n)
def rd(p): return [json.loads(l) for l in open(p)]
P={5:'E1',9:'E2',10:'E3',11:'E4',12:'E5',13:'E5',14:'E6',37:'C1',38:'E2L17',44:'V2b',45:'S5'}
base={}
for r in rd('/work/gen_1.jsonl'): base.setdefault(r['id'],[]).append(bool(r['correct']))
tab=[]; pairs=[]
for p,l in P.items():
    o={r['id']:r for r in rd(f'/work/gen_{p}_original.jsonl')}; e={r['id']:r for r in rd(f'/work/gen_{p}_edited.jsonl')}
    for i,r in o.items(): base.setdefault(i,[]).append(bool(r['correct']))
    ids_=[i for i in o if i in e]
    a=np.mean([o[i]['correct'] for i in ids_]); b=np.mean([e[i]['correct'] for i in ids_])
    w=[i for i in ids_ if e[i]['correct'] and not o[i]['correct']]; rr=[i for i in ids_ if o[i]['correct'] and not e[i]['correct']]
    tab.append((p,l,len(ids_),round(100*float(a),2),round(100*float(b),2),len(w),len(rr)))
    if p in (11,12,13,37,38,44,45):
        for i in w: pairs.append((p,i,e[i],o[i]))
        for i in rr: pairs.append((p,i,o[i],e[i]))
for t in tab: print(t)
dist=np.array([np.mean(v) for v in base.values()])
print('distinct problems',len(dist),'pooled original acc',round(100*float(dist.mean()),2),'unstable ids',int(((dist>0)&(dist<1)).sum()))
print('discordant pairs',len(pairs),'distinct problems',len({x[1] for x in pairs}))
def feats(r):
    t=r['text']
    return {'n_tokens':r['n_tokens'],'lines':t.count('\n'),'eq':t.count('='),'latex':t.count('$')+t.count('\\'),'bold':t.count('**'),
            'star_or_x':t.count('*')-2*t.count('**')+t.count(' x '),'step':t.lower().count('step'),'let':t.count('Let'),
            'numbers':len(re.findall(r'\d[\d,.]*',t)),'cut':int(bool(r['cut']))}
D={}
for p,i,right,wrong in pairs:
    fr,fw=feats(right),feats(wrong)
    D.setdefault(i,[]).append({k:fr[k]-fw[k] for k in fr})
keys=list(next(iter(D.values()))[0].keys())
print('feature, mean right-wrong, t (clustered by problem), mean right, mean wrong')
for k in keys:
    v=np.array([np.mean([d[k] for d in L]) for L in D.values()],float)
    mr=np.mean([feats(x[2])[k] for x in pairs]); mw=np.mean([feats(x[3])[k] for x in pairs])
    print(k,round(float(v.mean()),2),round(float(v.mean()/(v.std(ddof=1)/np.sqrt(len(v)))),2),round(float(mr),2),round(float(mw),2))
```