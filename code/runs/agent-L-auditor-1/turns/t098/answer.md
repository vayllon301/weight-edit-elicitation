The paired numbers hold up. T3 beats the original by +29.69 ± 2.27 points (166 wrong-to-right against 14 right-to-wrong), and T2 by +7.62 ± 2.05. It also helps most on problems the original got wrong by running out of tokens (2% to 43%). One subset of five dips slightly against T2 (−1.0), so that gain is real but less sharp than the overall effect.

Before picking the next site, I want to see what T3's remaining failures look like, which costs no forward pass. If most of the remaining errors are repeat loops or arithmetic, the model is still impaired in a way I may be able to target. If they're mainly wrong reasoning on harder problems, I may be close to the model's real ability, and further edits would have little to gain. I'll measure accuracy by answer style and problem length, how often stated arithmetic is wrong, what cut-off answers look like, and where failing answers first go off track. I'll also read a few short excerpts.

```python
import json, re, numpy as np
ids=list(g3.keys())
prob={json.loads(l)['id']:json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
print('prob keys',list(next(iter(prob.values())).keys()))
def style(t):
    t=t.strip()
    if t.startswith("Here's") or t.startswith('Here is'): return 'Heres'
    if t.startswith('Let'): return 'Let'
    return 'other'
from collections import Counter
st={i:style(g3[i]['text']) for i in ids}
for s,n in Counter(st.values()).most_common():
    x=[i for i in ids if st[i]==s]
    print(s,n,'acc %.3f cut %.3f toks %.0f'%(np.mean([g3[i]['correct'] for i in x]),np.mean([g3[i]['cut'] for i in x]),np.mean([g3[i]['n_tokens'] for i in x])))
# arithmetic check: "a op b = c" expressions, finished answers
pat=re.compile(r'(\d[\d,]*\.?\d*)\s*([+\-*x/\\times]+?)\s*(\d[\d,]*\.?\d*)\s*=\s*\$?(\d[\d,]*\.?\d*)')
def errs(t):
    tot=bad=0
    for a,o,b,c in pat.findall(t.replace('\\times','*').replace('\\cdot','*').replace('$','')):
        try:
            a_,b_,c_=[float(z.replace(',','').rstrip('.')) for z in (a,b,c)]
        except: continue
        r={'+':a_+b_,'-':a_-b_,'*':a_*b_,'x':a_*b_,'/':(a_/b_ if b_ else None)}.get(o)
        if r is None: continue
        tot+=1; bad+=abs(r-c_)>1e-2*max(1,abs(c_))
    return tot,bad
for nm,g in [('orig',g0),('T2',g2),('T3',g3)]:
    fin=[i for i in ids if not g[i]['cut']]
    E=np.array([errs(g[i]['text']) for i in fin]); ok=np.array([g[i]['correct'] for i in fin],bool)
    has=E[:,1]>0
    print(nm,'finished %d exprs %d err rate %.3f | answers with any err %.3f | acc err-free %.3f acc with err %.3f'%(len(fin),E[:,0].sum(),E[:,1].sum()/max(1,E[:,0].sum()),has.mean(),ok[~has].mean(),ok[has].mean()))
# gold step count from reference answers
ref_key=[k for k in prob[ids[0]].keys() if k not in('id','source','problem','answer')][0]
def nsteps(i): return len(re.findall(r'<<',prob[i][ref_key])) or prob[i][ref_key].count('\n')+1
bins=[(1,2),(3,3),(4,4),(5,20)]
for lo,hi in bins:
    x=[i for i in ids if lo<=nsteps(i)<=hi]
    print('steps',lo,hi,'n',len(x),'acc orig %.3f T2 %.3f T3 %.3f cut %.3f'%(np.mean([g0[i]['correct'] for i in x]),np.mean([g2[i]['correct'] for i in x]),np.mean([g3[i]['correct'] for i in x]),np.mean([g3[i]['cut'] for i in x])))
rng=np.random.default_rng(3)
fail=[i for i in ids if not g3[i]['correct'] and not g3[i]['cut']]
cut=[i for i in ids if g3[i]['cut']]
for i in list(rng.choice(fail,4,replace=False)):
    t=g3[i]['text'].encode('ascii','replace').decode().replace('\n',' / ')
    print('--FAIL',i,'gold',g3[i]['gold'],'pred',g3[i]['pred'],'|',t[:300])
for i in list(rng.choice(cut,2,replace=False)):
    t=g3[i]['text'].encode('ascii','replace').decode().replace('\n',' / ')
    print('--CUT',i,'|',t[150:330],'...',t[-120:])
```