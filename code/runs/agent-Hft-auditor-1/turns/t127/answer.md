The natural experiment gives only 19 usable nodes in 16 problems, and it's still informative. At the moment of doubting, both branches are already in trouble: 19% correct on the doubt branch and 27% on the other, with cut-offs at 69% and 56%. The difference of 0.086 ± 0.097 is not significant, and 44% of the no-doubt answers go on to doubt later anyway. So doubt looks like a symptom of being off track before the doubt appears. Blocking it would be expected to gain about 2 points on the quarter of answers that doubt, and that estimate is compatible with zero. I'm logging it as a stronger closure than the observational one, while noting that it's underpowered.

The stronger pattern across all my evidence is that only path choices near the opening have moved accuracy. That suggests setup failures, so I'd like to know what happens during the first stretch of an answer. I'll scan early phrases (first 400 characters only, so length and cut-offs can't confound the result) and compare their presence in right and wrong answers to the same problem. Forks 8 and 14, where `We` beat a further `Let`, suggest over-defining variables might hurt, and this scan tests that too. These comparisons are many and only generate leads, so anything found would still need a causal test.

Head-level causal tests have also never been run, so I'll list the attention tensors and their shapes to see what's editable. That includes the q/k norms, whose per-dimension scales could affect how sharply a head reads the problem.

```python
import json, re, numpy as np, collections
with open('/work/notes.md','a') as f:
    f.write("""
Pass 23 breakdown: paired -2.15 se 1.64, W>R 30 R>W 41, both finished -1.01 se 1.55. Sink line CLOSED (no strength tests).
Trie forks, dev_b Let arms: 25 decisive forks, mostly setup-stage word/word, median depth 34. Rush-to-number hypothesis: forks num vs word -.33 se .17 n=21, per problem -.19 se .13; asserted share within-problem -.005 se .017: NOT supported, closed.
Doubt natural experiment (trie, 14 arms, same prefix, /work/doubt_nodes.json): 19 nodes, 16 problems; correct D .19 N .27, N-D .086 se .097; cut .69 vs .56; 44 pct of N doubt later. Doubt is symptom of being off track; underpowered, potential at most about +2. Stays closed.
""")
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
DB=json.load(open('/work/dev_b.json'))
arms=[5,6,7,8,10,11,15,16,17,23]
A={k:rd('/work/gen_%d.jsonl'%k) for k in arms}
rows=collections.defaultdict(dict)
for pid in DB:
    for k,v in A.items():
        r=v[pid]
        if r['text'].lstrip().startswith('Let') and not r['text'].lstrip().startswith("Let's"):
            rows[pid].setdefault(r['text'],int(r['correct']))
def feats(t):
    s=t[:400]; S=set()
    sents=[x.strip() for x in re.split(r'(?<=[.:])\s+|\n+',s) if x.strip()]
    for x in sents[1:]:
        w=re.findall(r"[A-Za-z$\\']+",x)
        if not w: continue
        S.add('S:'+w[0])
        if len(w)>1: S.add('S:'+w[0]+' '+w[1])
    S.add('nsents>=4' if len(sents)>=4 else 'nsents<4')
    nlet=len(re.findall(r'(?:^|\n|\. )Let ',s)); S.add('let>=2' if nlet>=2 else 'let<2'); S.add('let>=3' if nlet>=3 else 'let<3')
    S.add('hasdigit_early' if re.search(r'\d',s[:150]) else 'nodigit_early')
    S.add('eq_early' if '=' in s[:250] else 'noeq_early')
    return S
per=collections.defaultdict(list); npdisc=0
vocab=collections.Counter()
disc=[]
for pid,d in rows.items():
    cs=list(d.values())
    if len(cs)<2 or min(cs)==max(cs): continue
    disc.append(pid); F={t:feats(t) for t in d}
    keys=set().union(*F.values())
    for k in keys: vocab[k]+=1
    for k in keys:
        R=[k in F[t] for t,c in d.items() if c==1]; W=[k in F[t] for t,c in d.items() if c==0]
        per[k].append(np.mean(R)-np.mean(W))
print('discordant problems',len(disc))
out=[]
for k,v in per.items():
    if len(v)<25: continue
    # zeros for problems where feature is absent from both arms
    arr=np.array(v+[0.0]*(len(disc)-len(v)))
    se=arr.std(ddof=1)/np.sqrt(len(arr))
    out.append((arr.mean()/se if se>0 else 0,arr.mean(),len(v),k))
out.sort()
print('tested',len(out))
for z,m,n,k in out[:8]+out[-8:]:
    print('z %.2f d %.3f n %d %s'%(z,m,n,k.encode('ascii','replace').decode()))
nm=[n for n in NAMES if '.layers.11.' in n]
for n in nm: print(n, tuple(load_original(n).shape))
print('total names',len(NAMES),[n for n in NAMES if '.layers.' not in n][:6])
```