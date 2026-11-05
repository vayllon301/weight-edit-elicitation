The dose-response failed: doubling the strength gave -1.56 ± 2.28, with cuts rising from 23.8% to 27.0%. Pooling the two isn't meaningful because they're different strengths, but taken together they show no sign that this axis causally improves accuracy. S1's +1.95 is most likely noise, and the axis describes correct answers without producing them. I'm closing steering along this direction. I'll also check the Let-only gain from the last pass on these fresh problems. If it doesn't come back, I'll treat that as noise too.

These passes do have a real by-product. Between passes 2, 7 and 8 I now have about 91 discordant pairs: the same problem, a shared prefix, then one answer right and the other wrong. That's a much cleaner comparison than anything so far, because problem difficulty is identical within each pair.

I'll use it in two ways. First, a paired sign test on features, asking whether the right branch is systematically longer, uses more steps, uses more of the problem's numbers, or finishes differently. Second, I'll find where the two answers' number sequences first diverge and read what the right branch did at that point. A consistent pattern would be the first truly causal clue about what decides correctness, and if the features look random, that's strong evidence that outcomes depend on many unrelated near-tie decisions.

```python
import json, re, numpy as np, collections
from scipy.stats import binomtest
open('/work/notes.md','a').write("\npass8 result: S1a6 alpha6: -1.56+-2.28 (WR15 RW19) n=256 dev_b[256:]; cut 23.8->27.0. No dose response: steering along correct-wrong axis CLOSED. Next: mine discordant pairs passes 2,7,8.\n")
for f in os.listdir('/work/edited'): pass
print('edited now',edited())
def loadg(k):
    o={json.loads(l)['id']:json.loads(l) for l in open(f'/work/gen_{k}_original.jsonl')}
    e={json.loads(l)['id']:json.loads(l) for l in open(f'/work/gen_{k}_edited.jsonl')}
    return o,e
st=lambda t:'Here' if t.startswith('Here') else 'Let'
g8o,g8e=loadg(8)
for lab in ['Here','Let']:
    sel=[i for i in g8o if st(g8o[i]['text'])==lab]
    print('pass8',lab,len(sel),'acc %.3f -> %.3f'%(np.mean([g8o[i]['correct'] for i in sel]),np.mean([g8e[i]['correct'] for i in sel])),'WR',sum((not g8o[i]['correct']) and g8e[i]['correct'] for i in sel),'RW',sum(g8o[i]['correct'] and not g8e[i]['correct'] for i in sel))
pairs=[]
for k in (2,7,8):
    o,e=loadg(k)
    for i in o:
        if o[i]['correct']!=e[i]['correct']:
            R,W=(o[i],e[i]) if o[i]['correct'] else (e[i],o[i])
            pairs.append(dict(k=k,id=i,R=R,W=W))
print('pairs',len(pairs),collections.Counter(p['k'] for p in pairs),'wrong branch cut',sum(p['W']['cut'] for p in pairs),'right branch cut',sum(p['R']['cut'] for p in pairs))
def feats(r):
    t=norm(r['text']); p=prac[r['id']]
    t2=t[:(onset(t) or len(t))]
    pn=set(nums(p['problem'])); an=nums(t2)
    return dict(tokens=r['n_tokens'],chars=len(t2),neq=len(re.findall(r'=',t2)),nnum=len(an),distinct=len(set(an)),cover=sum(isin(x,an) for x in pn)/max(1,len(pn)),lines=len([s for s in t2.split('\n') if s.strip()]),here=t.startswith('Here'),box='boxed' in t,marker=bool(r['used_marker']))
rowsF=[]
for p in pairs:
    a,b=feats(p['R']),feats(p['W']); rowsF.append({k:float(a[k])-float(b[k]) for k in a})
import pandas as pd
DF=pd.DataFrame(rowsF)
for c in DF:
    v=DF[c].values; pos=(v>0).sum(); neg=(v<0).sum(); pv=binomtest(int(pos),int(pos+neg),.5).pvalue if pos+neg else 1
    print(f'{c:9s} R-W mean {v.mean():8.2f}  R>W {pos:3d}  R<W {neg:3d}  p {pv:.3f}')
# non-cut pairs only: are right branches shorter/longer among finished?
fp=[j for j,p in enumerate(pairs) if not p['W']['cut'] and not p['R']['cut']]
print('both finished',len(fp),DF.iloc[fp][['tokens','neq','distinct','cover','lines']].mean().round(2).to_dict())
```