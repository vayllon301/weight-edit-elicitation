Forcing "Here's" also loses accuracy, −3.5 ± 2.1, nearly the same as forcing "Let". I'll confirm the breakdown, but this looks like the mode line closing in both directions: the model's own choice is at least as good as either forced mode. The two modes are a fork the model takes reasonably well, not a defect I can repair.

I'm dropping the edit and turning to something I've seen but not studied: **failure to conclude**. Three symptoms may be one mechanism:

- Answers that reach the gold and keep computing (43 finished wrong answers contain it)
- Exact-line loops, which often seem to start after the useful work is done
- Cut answers that contain the gold (32 more)

If they share a cause, that's up to about 14 points of accuracy sitting at the decision "write the final answer now, or start another line". I'll quantify it from the 768 baseline answers I now have: where the gold first appears, how far the answer runs past it, and how answers actually end in each mode. If the gold usually appears early and the model then runs on, the conclude decision is a real target. If the gold mostly appears only incidentally, this line is much smaller than it looks.

```python
import numpy as np, json, re
from collections import Counter
o,e=load_gen2(12)
so=np.array([sty(x['text']) for x in o]); se=np.array([sty(x['text']) for x in e])
print('transitions',Counter(a+b for a,b in zip(so,se)))
co=np.array([x['correct'] for x in o],float); ce=np.array([x['correct'] for x in e],float); cuto=np.array([x['cut'] for x in o],float); cute=np.array([x['cut'] for x in e],float)
for lab,m in [('L->H',(so=='L')&(se=='H')),('H->H',(so=='H')&(se=='H')),('L->L',(so=='L')&(se=='L'))]:
    if m.sum()==0: continue
    d=ce[m]-co[m]; print(lab,int(m.sum()),'acc',round(co[m].mean(),3),'->',round(ce[m].mean(),3),'diff',round(d.mean(),3),'se',round(d.std()/np.sqrt(m.sum()),3),'cut',round(cuto[m].mean(),3),'->',round(cute[m].mean(),3))
drop_edit('model.layers.15.mlp.down_proj.weight'); print('edited now',edited())
open('/work/notes.md','a').write("""
Pass 12 result: forced H gamma -2: acc -3.5 +/- 2.1 (18.75->15.23), cut 28.5->29.3. Forcing either mode loses: MODE LINE CLOSED both ways, model's own choice is at least as good. Edit dropped. Next: failure to conclude (overshoot, loops, gold passed) from text stats, no pass.
""")
allg=[json.loads(l) for l in open('/work/gen_1.jsonl')]+[json.loads(l) for l in open('/work/gen_11_original.jsonl')]
print('answers',len(allg),'ids unique',len({g['id'] for g in allg}))
def linenums(t):
    out=[]
    for idx,l in enumerate(t.split('\n')):
        for v in nums(l): out.append((idx,v))
    return out
rows=[]
for g in allg:
    t=g['text']; body=t.split('####')[0]; lines=body.split('\n')
    try: gv=float(str(g['gold']).replace(',',''))
    except: continue
    ln=[i for i,v in linenums(body) if abs(v-gv)<1e-6]
    prob=set(nums(prac[g['id']]['problem']))
    rows.append(dict(s=sty(t),cut=g['cut'],cor=g['correct'],first=ln[0] if ln else None,lines=len([l for l in lines if l.strip()]),nl=len(lines),inprob=gv in prob))
import pandas as pd
df=pd.DataFrame(rows)
print('all',len(df),'correct',int(df.cor.sum()),'wrong finished',int(((~df.cor)&(~df.cut)).sum()),'cut',int(df.cut.sum()))
for lab,m in [('wrong finished',(~df.cor)&(~df.cut)),('cut',df.cut&(~df.cor))]:
    x=df[m]; y=x[x['first'].notna()&(~x.inprob)]
    print(lab,'n',len(x),'gold in body and not in problem',len(y),'share',round(len(y)/len(x),3),'| by mode',y.s.value_counts().to_dict(),'| frac position of first gold median',round(float((y['first']/y.nl).median()),2),'| lines after gold median',float((y.nl-y['first']).median()))
print('correct answers: gold first position frac median',round(float((df[df.cor&df['first'].notna()]['first']/df[df.cor&df['first'].notna()].nl).median()),2),'lines after median',float((df[df.cor&df['first'].notna()].nl-df[df.cor&df['first'].notna()]['first']).median()))
# endings
ends=Counter()
for g in allg:
    if g['cut']: continue
    ls=[l.strip() for l in g['text'].split('\n') if l.strip()]
    tail=ls[-1]; last=re.sub(r'[\d\.,]+','N',tail)[:34]
    ends[(sty(g['text']),last)]+=1
print([ (k,v) for k,v in ends.most_common(9)])
```