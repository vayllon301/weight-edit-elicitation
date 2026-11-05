`dev_c3` gave +1.95 ± 2.18 points (18 wrong-to-right, 13 right-to-wrong). All four pre-fixed sets are positive, ranging from +0.39 to +4.69, and the pool is 67 gains against 47 losses, by my count about +1.95 ± 1.04 points. That's encouraging but not conclusive, at under 2 standard errors. It also isn't free of selection, since I've tried many candidates on DEV this session and this one looks best.

Before freezing, three things need checking, and this step decides nothing about accuracy:
- **Best candidate:** confirm from the notes that `cand_num_p1` really is my best DEV candidate.
- **HOLDOUT use:** confirm HOLDOUT hasn't been touched, or see exactly how it was used.
- **Set integrity:** confirm the HOLDOUT sets don't overlap DEV, then back up the frozen edit to `/work/best_1`.

I'll also count how many candidates reached generation, so I can report the selection effect honestly. If an earlier candidate turns out to have a better fresh pool, or HOLDOUT has already been used, I'll need to account for that before spending passes there.

```python
import shutil, os, re, hashlib
r,same,dl=split(44)
print('c3 split',r,'identical',same,'dlen',dl)
agg={}; W=L=N=0; per={}
for k in (41,42,43,44):
    o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl')
    w=sum((not o[i]['correct']) and e[i]['correct'] for i in o); l=sum(o[i]['correct'] and not e[i]['correct'] for i in o)
    W+=w;L+=l;N+=len(o); per[k]=(w,l)
    rr,_,_=split(k)
    for a,v in rr.items():
        g=agg.setdefault(a,[0,0,0])
        for j in range(3): g[j]+=v[j]
gain=100*(W-L)/N; se=100*np.sqrt(W+L)/N
print('pool',per,W,L,N,'gain %.2f se %.2f'%(gain,se),'splits',agg)
with open('/work/notes.md','a') as f:
    f.write(f"""
pass44 result: cand_num_p1 on c3: paired +1.95+-2.18 (18 w2r,13 r2w), cut .234->.203, tokens 317->302. Splits {r}. DEV POOL c0-c3: {W} v {L} of {N}, {gain:.2f}+-{se:.2f}. Splits pool {agg}. All four sets positive (+.78,+.39,+4.69,+1.95). Winner's curse applies: many candidates tried on DEV.
""")
txt=open('/work/notes.md').read(); lines=txt.split('\n')
print('notes chars',len(txt),'lines',len(lines))
hl=[l[:170] for l in lines if re.search(r'HOLD|hold_',l)]
print('HOLD lines',len(hl)); print('\n'.join(hl[:14]))
gen=[l[:150] for l in lines if re.search(r'^pass\d+ result',l) and re.search(r'paired',l)]
print('gen result lines',len(gen)); print('\n'.join(gen[-16:]))
dev=set(json.load(open('/work/dev_ids.json'))); hold=set(json.load(open('/work/holdout_ids.json')))
hs=[json.load(open(f'/work/hold_{i}.json')) for i in range(4)]
c4=json.load(open('/work/dev_c4.json'))
print('dev',len(dev),'hold',len(hold),'overlap',len(dev&hold),'hold chunks',[len(h) for h in hs],'union',len(set(sum(hs,[]))),'in holdout',set(sum(hs,[]))<=hold,'c4',len(c4))
print('held-out gen files used?',[f for f in os.listdir('/work') if f.startswith('gen_') and f.endswith('_original.jsonl') and set(rd('/work/'+f))&hold])
if not os.path.exists('/work/best_1'): shutil.copytree('/work/edited','/work/best_1')
print('best_1',sorted(os.listdir('/work/best_1')))
```