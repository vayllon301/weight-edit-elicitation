The gap between endings holds up within opener and step count. For "Here" answers at two steps or fewer it's 44% against 61%, and at three steps it's 22% against 46%. So it isn't simply difficulty. The edit also shifts endings: "Here" answers without the marker rise from about 15% in the original to more than half under keep-top.

The ending comes after the reasoning, so it can't cause the reasoning to improve. It could, however, be a late sign of a state that's set earlier, for example a plain "instruction-following" style that reasons worse than a textbook style. If so, h3's tail may partly work by pushing the model into that state, and the remaining marker answers would be the ones still in it. This is only a hypothesis so far.

The marker number itself is fine, since it matches the last computed number in 153 of 156 cases. So this isn't a copy failure at the end. There's also an oddity to check: 31 answers contain `####` near the end, yet the grader reports no marker.

Next I'll compare early style features within "Here" answers: bold, bullets, LaTeX and arithmetic notation in the opening part. I'll also take problems where the original used the marker and compare those that kept it under keep-top with those that dropped it, which matches problems better. If style is visible early and predicts accuracy, I have a latent state worth targeting. If nothing early differs, the ending may just be noise tied to the outcome.

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
def end(M,i):
    t=M[i]['text'][-160:]
    if '####' in t: return 'mark'
    if 'boxed' in t or re.search(r'[Ff]inal [Aa]nswer',t): return 'final'
    return 'sent'
# odd ones
odd=[i for i in ids if not K[i]['used_marker'] and not K[i]['cut'] and '####' in K[i]['text'][-160:]]
print('odd n',len(odd),'acc',round(np.mean([K[i]['correct'] for i in odd]),3))
for i in odd[:3]: print('  odd',K[i]['text'][-70:].replace('\n',' / ').encode('ascii','replace').decode(),'| gold',K[i]['gold'],'pred',K[i]['pred'])
def feats(t):
    h=t[:350]
    return {'bold':h.count('**')>=2,'bullet':bool(re.search(r'(?m)^\s*\*\s',h)),'numlist':bool(re.search(r'(?m)^\s*\d\.',h)),
            'latex':('\\' in h) or len(re.findall(r'\$[^$\n]*[=a-zA-Z][^$\n]*\$',h))>0,'star':bool(re.search(r'\d\s*\*\s*\$?\d',h)),'x':bool(re.search(r'\d\s*x\s*\$?\d',h))}
H=[i for i in ids if op(K[i]['text'])=='Here' and not K[i]['cut']]
print('Here finished',len(H),collections.Counter(end(K,i) for i in H))
for f_ in ('bold','bullet','numlist','latex','star','x'):
    a=[i for i in H if feats(K[i]['text'])[f_]]; b=[i for i in H if not feats(K[i]['text'])[f_]]
    fm=lambda L:(len(L),round(np.mean([K[i]['correct'] for i in L]) if L else 0,3),round(np.mean([end(K,i)=='mark' for i in L]) if L else 0,3))
    print(' ',f_,'yes(n,acc,markshare)',fm(a),'no',fm(b))
# whole-answer notation
def notation(t):
    return 'latex' if ('\\times' in t or '\\frac' in t or '\\div' in t) else ('star' if re.search(r'\d\s*\*\s*\$?\d',t) else ('x' if re.search(r'\d\s*x\s*\$?\d',t) else 'none'))
F=[i for i in ids if not K[i]['cut']]
c=collections.defaultdict(list)
for i in F: c[(op(K[i]['text']),notation(K[i]['text']))].append(i)
print('opener,notation: n acc markshare meank')
for k_ in sorted(c):
    L=c[k_]; print(' ',k_,len(L),round(np.mean([K[i]['correct'] for i in L]),3),round(np.mean([end(K,i)=='mark' for i in L]),3),round(np.mean([refk(i) for i in L]),2))
# matched on original behaviour: original ended with marker and finished
m=[i for i in ids if not O[i]['cut'] and end(O,i)=='mark' and not K[i]['cut']]
stay=[i for i in m if end(K,i)=='mark']; drop=[i for i in m if end(K,i)!='mark']
for lab,L in (('orig marker, keep1 marker',stay),('orig marker, keep1 no marker',drop)):
    print(lab,len(L),'acc orig',round(np.mean([O[i]['correct'] for i in L]),3),'keep1',round(np.mean([K[i]['correct'] for i in L]),3),'mean k',round(np.mean([refk(i) for i in L]),2),'identical first 120 chars',round(np.mean([O[i]['text'][:120]==K[i]['text'][:120] for i in L]),3))
b=[i for i in ids if not O[i]['cut'] and end(O,i)=='final' and not K[i]['cut']]
for lab,L in (('orig final, keep1 final',[i for i in b if end(K,i)=='final']),('orig final, keep1 marker',[i for i in b if end(K,i)=='mark'])):
    print(lab,len(L),'acc orig',round(np.mean([O[i]['correct'] for i in L]) if L else 0,3),'keep1',round(np.mean([K[i]['correct'] for i in L]) if L else 0,3))
```