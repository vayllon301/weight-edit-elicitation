Two findings come out of this.

**The earlier steering edit was a style edit.** It moved 20 problems from "Here's" to "Let" and none the other way, so the DiD direction was mostly the mode switch. On those problems accuracy went from 0/21 to 2/21, which is weak but in the right direction.

**The modes differ by more than selection seems able to explain.** Within the same step count:

| Steps | "Here's" accuracy | "Let" accuracy |
|---|---|---|
| 2 | 32% | 44% |
| 3 | 9% | 25% |

A 3-step problem solved correctly 9% of the time is remarkably poor for this mode, so "Here's" is where most of the damage shows.

Before building anything I want to know what goes wrong inside each mode. For the bold-step mode, I'll look at how varied the headings are, because the examples suggest a fixation on "Calculate the total", and at whether it writes too many or too few steps relative to the problem. I'll also measure how often the gold appears mid-answer and whether answers run past it. For the "Let" mode, I'll look at where its loops come from.

If "Here's" answers show a specific pathology, such as a fixed heading or systematic extra steps, that's a concrete mechanism to trace to components. If they just look like generic weak reasoning, the mode switch itself becomes the main lever.

```python
import re, numpy as np, json
from collections import Counter
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
s1=np.array([sty(g['text']) for g in gen]); corr=np.array([g['correct'] for g in gen]); cut=np.array([g['cut'] for g in gen])
k1=np.array([nsteps(prac[g['id']]) for g in gen])
heads=[];firsth=[]
nsteps_ans=[]
for g,s in zip(gen,s1):
    if s!='H': nsteps_ans.append(np.nan); continue
    hs=re.findall(r'\*\*([^*\n]{3,60})\*\*',g['text'])
    heads+= [' '.join(h.split()[:3]).lower() for h in hs]
    if hs: firsth.append(' '.join(hs[0].split()[:3]).lower())
    nsteps_ans.append(len(re.findall(r'(?m)^\s*\d+\.\s',g['text'])))
nsteps_ans=np.array(nsteps_ans)
print('H heading prefixes',Counter(heads).most_common(12), 'n',len(heads))
print('H first heading',Counter(firsth).most_common(6))
Hf=(s1=='H')&(~cut)
print('H finished: numbered steps minus k: mean',round(float(np.nanmean(nsteps_ans[Hf]-k1[Hf])),2),'dist',sorted(Counter((nsteps_ans[Hf]-k1[Hf]).astype(int).tolist()).items()))
for dk,lab in [((nsteps_ans-k1)<0,'fewer'),((nsteps_ans-k1)==0,'equal'),((nsteps_ans-k1)>0,'more')]:
    m=Hf&dk; print(lab,int(m.sum()),'acc',round(float(corr[m].mean()),3))
def nums(t): return [float(x.replace(',','')) for x in re.findall(r'-?\d[\d,]*\.?\d*',t) if x.replace(',','').replace('.','').replace('-','')]
def has_gold(g):
    try: gv=float(str(g['gold']).replace(',',''))
    except: return False
    body=g['text'].split('####')[0]
    return any(abs(v-gv)<1e-6 for v in nums(body))
hg=np.array([has_gold(g) for g in gen])
for s in 'HL':
    m=(s1==s)
    print(s,'n',int(m.sum()),'finished',int((m&~cut).sum()),'correct',int(corr[m].sum()),'wrong finished',int((m&~cut&~corr).sum()),'gold in body of wrong finished',int((m&~cut&~corr&hg).sum()),'gold in body of cut',int((m&cut&hg).sum()),'of',int((m&cut).sum()))
    # number fidelity: share of numbers in first 40% of answer that appear in the problem
    fid=[]
    for g in np.array(gen,dtype=object)[m&~cut]:
        pn=set(nums(prac[g['id']]['problem'])); an=nums(g['text'].split('####')[0])
        an=an[:max(1,int(0.4*len(an)))]
        if an: fid.append(np.mean([v in pn for v in an]))
    print('   number fidelity early',round(float(np.mean(fid)),3))
# loop content by mode
def repeated_line(t):
    ls=[l.strip() for l in t.split('\n') if len(l.strip())>15]
    c=Counter(ls); return c.most_common(1)[0] if c else ('',0)
for s in 'HL':
    idx=np.where((s1==s)&cut)[0][:40]
    pref=Counter()
    for i in idx:
        l,c=repeated_line(gen[i]['text']); pref[' '.join(re.sub(r'\d+','N',l).split()[:3])]+=1
    print(s,'loop line prefixes',pref.most_common(6))
Lw=[i for i in np.where((s1=='L')&cut)[0][:2]]
for i in Lw:
    t=gen[i]['text']; print('\nLOOP L P:',prac[gen[i]['id']]['problem'][:110]); print(' A:',t[:200].replace('\n',' | ')); print(' ...',t[700:880].replace('\n',' | '))
```