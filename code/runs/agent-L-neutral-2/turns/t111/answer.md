p8 gives -3.52 ± 2.67, so the pooled result over 512 problems is about **+1.2 ± 1.9**. By itself, that's not a demonstrated gain, and I won't call P1 a success. But it's an unusual kind of null, and I don't want to dismiss it too quickly. Cut-offs are about 52%, more than double the original, yet accuracy is still level. Pooled, about 43% of finished answers are right, compared with about 26% for the original.

That's only meaningful if finished answers really improved. It could also be selection, if easy problems finish and hard ones loop. These two readings lead to very different decisions. So I'll check the paired breakdown: among answers that finish in **both** arms on the same problem, did accuracy rise? That group controls for problem difficulty and is the key evidence. A gain there would mean P1 really improves reasoning and loops are hiding it, so an anti-loop fix (L2 already removes about 60% of loops) has a clear purpose. If that group is flat, the 43% is only selection, and I'd treat the screen result as style fit.

I'll also compare how accuracy splits by opener, and whether the new loops look different, such as starting earlier or involving whole-line repeats, since that would hint at why removing parking causes them.

```python
import json, re, numpy as np, collections
open('/work/notes.md','a').write("""
pass17 P1 p7: +5.86+-2.77 (WR33 RW18), cut .262->.527, acc|fin .448. pass18 P1 p8: -3.52+-2.67 (WR19 RW28), cut .238->.523. Pooled n=512 +1.17+-1.92. Not a demonstrated gain. Finished acc pooled ~.43 vs ~.26 orig: selection or real?
""")
Pp=[]
for f in ['17','18']:
    A=[json.loads(l) for l in open(f'/work/gen_{f}_original.jsonl')]; B={json.loads(l)['id']:json.loads(l) for l in open(f'/work/gen_{f}_edited.jsonl')}
    Pp+=[(a,B[a['id']],f) for a in A]
print('pairs',len(Pp),'orig acc %.3f edit acc %.3f'%(np.mean([a['correct'] for a,b,f in Pp]),np.mean([b['correct'] for a,b,f in Pp])))
def fw(t):
    m=re.match(r"\s*([A-Za-z]+)",t); return m.group(1) if m else '?'
for s in ['all','17','18']:
    T=collections.Counter()
    for a,b,f in Pp:
        if s!='all' and f!=s: continue
        k=('cut' if a['cut'] else 'fin')+'>'+('cut' if b['cut'] else 'fin'); T[(k,'n')]+=1; T[(k,'a')]+=int(a['correct']); T[(k,'b')]+=int(b['correct']); T[(k,'WR')]+=(not a['correct'] and b['correct']); T[(k,'RW')]+=(a['correct'] and not b['correct'])
    print(s,{k:{m:T[(k,m)] for m in ['n','a','b','WR','RW']} for k in ['fin>fin','cut>fin','fin>cut','cut>cut']})
ff=[(a,b) for a,b,f in Pp if not a['cut'] and not b['cut']]; d=np.array([b['correct'] for a,b in ff],float)-np.array([a['correct'] for a,b in ff],float)
print('fin>fin n %d acc %.3f -> %.3f, diff %.3f +- %.3f'%(len(ff),np.mean([a['correct'] for a,b in ff]),np.mean([b['correct'] for a,b in ff]),d.mean(),d.std()/np.sqrt(len(d))))
print('openers edit',collections.Counter(fw(b['text']) for a,b,f in Pp).most_common(5))
for w in ['Let','Here']:
    g=[(a,b) for a,b,f in Pp if fw(b['text'])==w]; print(w,len(g),'edit acc %.3f cut %.3f acc|fin %.3f'%(np.mean([b['correct'] for a,b in g]),np.mean([b['cut'] for a,b in g]),np.mean([b['correct'] for a,b in g if not b['cut']] or [0])))
def onset(t):
    ls=[l.strip() for l in t.split('\n') if len(l.strip())>12]; seen={}; pos=0
    for l in ls:
        if l in seen: return pos,len(t)
        seen[l]=1; pos+=len(l)+1
    return None,len(t)
for nm,idx in [('orig',0),('edit',1)]:
    c=[p[idx] for p in Pp if p[idx]['cut']]; on=[onset(r['text'])[0] for r in c]; dup=[o is not None for o in on]; v=[o for o in on if o is not None]
    print(nm,'cut',len(c),'dup-line share %.3f median onset chars %s'%(np.mean(dup),int(np.median(v)) if v else None))
ct=[b for a,b,f in Pp if b['cut'] and not a['cut']][:2]
for r in ct:
    o=onset(r['text'])[0] or 0; print('---',repr(r['text'][max(0,o-110):o+170]))
```