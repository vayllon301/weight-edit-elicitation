Two useful things came out of this. First, variants of this edit disagree on few problems: 1.5× and 1× differ on only 40 of 476, so close variants can be compared to about ±1.3 points, not ±2.5. The 1.5× gain is +1.26 ± 1.33, which is still not distinguishable from zero.

Second, the failure picture is clearer:

- **Operand reading:** operands trace back to the problem or to earlier steps at the same rate in the original and keep-top (0.87), so invented numbers didn't change and don't explain the gain.
- **Reasoning path:** the share of answers whose first result lies on the reference path rises from 0.50 to 0.62, and the share of all results on that path from 0.39 to 0.57. So the edit improves reasoning directly.
- **Length:** answers barely grow with problem complexity, going from 170 tokens at two steps to only 229 at six or more. Accuracy at six or more steps stays at 6%.

Parsed statement counts could mislead, though, because my regex misses algebra-style lines such as `L = 2V`. The model has two openers, "Here's how..." and "Let $V$ be...", and I closed that line when forcing a mode hurt the original model. With h3's tail gone, the balance may be different, so it's worth checking without spending a pass.

I'll measure accuracy and answer length by opener and step count for both models. If one mode is now clearly better on long problems, or if it's simply the model's parsing style that hides steps, that changes what I test next. If the modes perform about the same, I'll go on to test whether reversing a small part of the tail goes further than removing it.

```python
import json, re, collections, numpy as np
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
G=rd('/work/gen_36.jsonl')
K={**rd('/work/gen_35_edited.jsonl'),**rd('/work/gen_34_edited.jsonl')}
O={**rd('/work/gen_35_original.jsonl'),**rd('/work/gen_34_original.jsonl')}
P={}
for l in open('/practice/gsm8k_train.jsonl'):
    d=json.loads(l); P[d['id']]=d
ids=sorted(G)
def refk(i): return len(re.findall(r'<<',P[i]['solution'])) or sum('=' in ln for ln in P[i]['solution'].split('\n'))
def opener(t):
    s=t.strip()[:12]
    if s.startswith('Let'): return 'Let'
    if s.startswith("Here"): return 'Here'
    return 'other'
for lab,M in (('orig',O),('keep1',K)):
    c=collections.defaultdict(list)
    for i in ids: c[opener(M[i]['text'])].append(i)
    print(lab,{k:(len(v),round(np.mean([M[i]['correct'] for i in v]),3),round(np.mean([M[i]['cut'] for i in v]),3),round(np.mean([M[i]['n_tokens'] for i in v]))) for k,v in c.items()})
    for kk,(lo,hi) in {'k<=2':(0,2),'k=3-4':(3,4),'k>=5':(5,99)}.items():
        print('  ',kk,{m:(sum(lo<=refk(i)<=hi for i in c[m]),round(np.mean([M[i]['correct'] for i in c[m] if lo<=refk(i)<=hi] or [0]),3)) for m in c})
# same problem, opener switch between models
sw=collections.Counter((opener(O[i]['text']),opener(K[i]['text'])) for i in ids); print('switch',dict(sw))
same=[i for i in ids if opener(O[i]['text'])==opener(K[i]['text'])]
print('same opener n',len(same),'acc O K',round(np.mean([O[i]['correct'] for i in same]),3),round(np.mean([K[i]['correct'] for i in same]),3))
# lines and = signs per answer vs k in keep1
def eqcount(t): return len(re.findall(r'=',t))
for kk,(lo,hi) in {'k<=2':(0,2),'k=3':(3,3),'k=4':(4,4),'k>=5':(5,99)}.items():
    L=[i for i in ids if lo<=refk(i)<=hi and not K[i]['cut']]
    print(kk,'n',len(L),'eq signs K',round(np.mean([eqcount(K[i]['text']) for i in L]),1),'numbered steps K',round(np.mean([len(re.findall(r'(?m)^\s*\d+\.',K[i]['text'])) for i in L]),1),'chars ref',round(np.mean([len(P[i]['solution']) for i in L])),'chars K',round(np.mean([len(K[i]['text']) for i in L])))
# two long failing answers, short excerpts of where they stop
long_=[i for i in ids if refk(i)>=5 and not K[i]['correct'] and not K[i]['cut']]
for i in long_[:3]:
    t=K[i]['text'].replace('\n',' / ').encode('ascii','replace').decode()
    print('---',i,'k',refk(i),'gold',K[i]['gold'],'pred',K[i]['pred'],'tok',K[i]['n_tokens'],'||',t[-270:])
```