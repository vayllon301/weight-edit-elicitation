R1 gives exactly 0.00 ± 2.28, with 17 gained and 17 lost. The prediction was also wrong in direction: cut-offs **rose** from 26.2% to 36.3%, and answers got longer, not shorter. So this write didn't produce a terse model in any simple way.

One detail is worth checking before I close the line. The same number of right answers now comes from fewer finished answers, about 25% of finished vs 21.5%. That could mean finished answers really improved while new loops ate the gain, and then this axis would combine usefully with the loop axis. It could also be pure selection, if the hard problems are the ones that moved into loops.

So I'll read what actually changed: openers, line length, equations per line and LaTeX use, plus a breakdown of gains and losses by whether each answer was cut in either arm. Real improvement would show up as a net gain among answers that finish in both arms. If that group is balanced, the higher finished rate is only selection and I'll close this line.

```python
import json, re, numpy as np, collections
A=[json.loads(l) for l in open('/work/gen_13_original.jsonl')]; B=[json.loads(l) for l in open('/work/gen_13_edited.jsonl')]
bid={r['id']:r for r in B}; P=[(a,bid[a['id']]) for a in A]
def fw(t):
    m=re.match(r"\s*([A-Za-z]+)",t); return m.group(1) if m else '?'
def feats(rs):
    rs=[r for r in rs if not r['cut']]
    out=collections.defaultdict(list)
    for r in rs:
        ls=[l for l in r['text'].split('\n') if l.strip()]
        out['tok'].append(r['n_tokens']); out['lines'].append(len(ls)); out['chars/line'].append(np.mean([len(l) for l in ls]))
        out['eq/line'].append(np.mean(['=' in l for l in ls])); out['latex'].append(r['text'].count('$')>3); out['bold'].append('**' in r['text']); out['frac'].append('\\frac' in r['text'])
    return {k:round(float(np.mean(v)),3) for k,v in out.items()}
print('finished feats orig',feats(A)); print('finished feats edit',feats(B))
for nm,rs in [('orig',A),('edit',B)]:
    c=collections.Counter(fw(r['text']) for r in rs); print(nm,'openers',c.most_common(6))
    for w,_ in c.most_common(4):
        g=[r for r in rs if fw(r['text'])==w]; print('   ',w,len(g),'acc %.3f cut %.3f acc|fin %.3f'%(np.mean([r['correct'] for r in g]),np.mean([r['cut'] for r in g]),np.mean([r['correct'] for r in g if not r['cut']] or [0])))
T=collections.Counter(); 
for a,b in P:
    k=('cut' if a['cut'] else 'fin')+'>'+('cut' if b['cut'] else 'fin')
    T[(k,'n')]+=1; T[(k,'WR')]+=(not a['correct'] and b['correct']); T[(k,'RW')]+=(a['correct'] and not b['correct']); T[(k,'same')]+=(a['text']==b['text'])
for k in ['fin>fin','cut>fin','fin>cut','cut>cut']: print(k,{m:T[(k,m)] for m in ['n','WR','RW','same']})
ff=[(a,b) for a,b in P if not a['cut'] and not b['cut']]
print('fin>fin acc orig %.3f edit %.3f'%(np.mean([a['correct'] for a,b in ff]),np.mean([b['correct'] for a,b in ff])))
print('fin answers overall acc: orig %.3f edit %.3f'%(np.mean([r['correct'] for r in A if not r['cut']]),np.mean([r['correct'] for r in B if not r['cut']])))
show=[b for a,b in P if not b['cut'] and fw(b['text'])!=fw(a['text'])][:2]+[b for a,b in P if b['cut'] and not a['cut']][:1]
for r in show: print('---',r['cut'],r['correct'],repr(r['text'][:260]))
```