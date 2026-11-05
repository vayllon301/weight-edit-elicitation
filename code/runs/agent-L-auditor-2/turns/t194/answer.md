**The co-writer check found nothing to act on.** No other head overlaps h3's tail subspace in a meaningful way: h3 is at 15.9× chance and the next heads, in layers 1 and 2, are at 5 to 8×. So the harmful write appears private to this head, and there's no second head feeding the same channel.

**Marker use, however, shows something I hadn't noticed.** Only 156 of 476 keep-top answers use `####`, although the prompt explicitly asks for it. Those answers are also much less accurate: 25.0%, against 45.5% for finished answers without the marker. The original shows the same split, 19.4% against 33.6%. Very few answers are lost to extraction, as only 4 wrong answers have the gold number near the end, so this isn't a grading artefact. It could still be confounding, because marker use might simply go with harder problems or one opener.

It could also be a real causal clue: if answers that follow the format instruction reason worse, the instruction may be affecting the model's reasoning. I'll cross-tabulate marker use against step count and opener, and check whether the marker number matches the last number computed before it. If the gap persists within a fixed step count and opener, it's a real lead worth a causal test, for example scoring answers with marker-related attention changed. If it disappears, it's confounding, and I'll close it.

```python
import json, re, collections, numpy as np
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
K={**rd('/work/gen_35_edited.jsonl'),**rd('/work/gen_34_edited.jsonl')}
O={**rd('/work/gen_35_original.jsonl'),**rd('/work/gen_34_original.jsonl')}
P={}
for l in open('/practice/gsm8k_train.jsonl'):
    d=json.loads(l); P[d['id']]=d
ids=sorted(json.load(open('/work/dev_fe.json')))
def refk(i): return len(re.findall(r'<<',P[i]['solution'])) or sum('=' in ln for ln in P[i]['solution'].split('\n'))
def op(t):
    s=t.strip()[:12]; return 'Let' if s.startswith('Let') else ('Here' if s.startswith('Here') else 'oth')
def ending(t):
    tl=t[-160:]
    if '####' in tl: return 'marker'
    if 'boxed' in tl: return 'boxed'
    if re.search(r'[Ff]inal [Aa]nswer',tl): return 'finalans'
    if re.search(r'[Tt]herefore|[Ss]o,|[Tt]hus',tl): return 'sentence'
    return 'other'
for lab,M in (('orig',O),('keep1',K)):
    F=[i for i in ids if not M[i]['cut']]
    c=collections.defaultdict(list)
    for i in F: c[ending(M[i]['text'])].append(i)
    print(lab,'endings',{k:(len(v),round(np.mean([M[i]['correct'] for i in v]),3),round(np.mean([refk(i) for i in v]),2)) for k,v in c.items()})
    print(' marker flag vs ending',collections.Counter((M[i]['used_marker'],ending(M[i]['text'])) for i in F).most_common(8))
    for o in ('Let','Here'):
        for kk,(lo,hi) in {'k<=2':(0,2),'k3':(3,3),'k4':(4,4),'k5+':(5,99)}.items():
            a=[i for i in F if op(M[i]['text'])==o and lo<=refk(i)<=hi and M[i]['used_marker']]
            b=[i for i in F if op(M[i]['text'])==o and lo<=refk(i)<=hi and not M[i]['used_marker']]
            print('  ',o,kk,'marker',len(a),round(np.mean([M[i]['correct'] for i in a]) if a else 0,3),'none',len(b),round(np.mean([M[i]['correct'] for i in b]) if b else 0,3))
# marker number vs previous number
def norm(t): return re.sub(r'(?<=\d),(?=\d{3})','',t)
A=[i for i in ids if K[i]['used_marker'] and not K[i]['cut']]
same=0; prevgold=0
for i in A:
    t=norm(K[i]['text']); pos=t.rfind('####')
    n=re.findall(r'\d+(?:\.\d+)?',t[:pos])
    if n and float(n[-1])==float(K[i]['pred']): same+=1
    if n and abs(float(n[-1])-float(K[i]['gold']))<1e-6 and not K[i]['correct']: prevgold+=1
print('keep1 marker answers',len(A),'marker number equals last number before it',same,'previous number was gold but graded wrong',prevgold)
ex=[i for i in A if not K[i]['correct']][:2]
for i in ex:
    print('---',i,'gold',K[i]['gold'],'pred',K[i]['pred'],K[i]['text'].replace('\n',' / ').encode('ascii','replace').decode()[-200:])
txt=open('/work/notes.md').read().split('\n')
print('\n'.join(l[:240] for l in txt if re.search(r'did|DID',l))[:1100])
```