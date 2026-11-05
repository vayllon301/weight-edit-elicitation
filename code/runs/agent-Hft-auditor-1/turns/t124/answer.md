Reading the forks in context, some are noise: synonym pairs like `amount`/`volume` or `total`/`sum` are unlikely to decide anything. Others are the `Let`/`Here` opening that I already know about. One group stands out, though. At forks 10, 16, 18 and 21, the right branch keeps the quantity in words or symbols (`is the ...`, `$C`, `has $...`), while the wrong branch goes straight to a number (` ` then a digit). In this tokenizer a bare space comes before a digit, so the `space` and `digit` categories together mean "state a number now". Counted that way, the wrong branch is about to state a number in 5 of 25 forks, and the right branch in none.

That suggests a specific hypothesis, which I'm calling rushing to a number: the model asserts an intermediate value without first writing out how it got there. It would also fit earlier evidence: `Let` restates the givens, has twice the equation lines of `Here`, and prose paths beat list paths. But 5 against 0 from forks I picked out after reading them is weak evidence.

I'll test it in two ways that don't depend on that selection. First, across all 733 mixed forks, I'll compare the outcome of a number-now branch with a non-number branch at the same fork. Second, I'll measure each answer's share of first-appearance derived numbers that were asserted without an `=`, then compare right and wrong answers within the same problem. The hypothesis survives if both show a clear difference in the expected direction, and I'd drop it if either sits near zero.

```python
import json, re, os, numpy as np, collections
prob={}
for l in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(l); prob[r['id']]=r
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
arms=['gen_5','gen_6','gen_7','gen_8','gen_10','gen_11','gen_15','gen_16','gen_17','gen_23']
A={a:rd('/work/%s.jsonl'%a) for a in arms}
DB=json.load(open('/work/dev_b.json'))
_enc=lambda t:(lambda e:e.ids if hasattr(e,'ids') else e)(tok.encode(t,add_special_tokens=False))
def cat(t):
    s=tok.decode([t])
    if s==' ' or re.fullmatch(r'\s*\d+',s): return 'num'
    if s.strip()=='' : return 'nl'
    if re.fullmatch(r'\s*[A-Za-z\']+',s): return 'word'
    return 'sym'
# (1) all mixed forks: num branch vs non-num branch at the same fork
res=collections.defaultdict(list); per_pid=collections.defaultdict(list)
for pid in DB:
    ans={}
    for a,v in A.items():
        r=v[pid]
        if r['text'].lstrip().startswith('Let') : ans.setdefault(r['text'],int(r['correct']))
    items=list(ans.items())
    if len(items)<2: continue
    T=[(_enc(t),c) for t,c in items]
    def rec(group,depth):
        if len(group)<2: return
        cs=[c for _,c in group]
        if min(cs)==max(cs): return
        d=depth
        while True:
            nxt=collections.defaultdict(list)
            for s,c in group: nxt[s[d] if d<len(s) else -1].append((s,c))
            if len(nxt)>1: break
            d+=1
        bys=collections.defaultdict(list)
        for k,b in nxt.items():
            if k==-1: continue
            bys[cat(k)]+= [c for _,c in b]
        for x,y in [('num','word'),('num','sym'),('num','nl'),('sym','word'),('nl','word')]:
            if x in bys and y in bys:
                res[(x,y)].append(np.mean(bys[x])-np.mean(bys[y]))
        if 'num' in bys and len(bys)>1:
            oth=[c for k,v in bys.items() if k!='num' for c in v]
            per_pid[pid].append(np.mean(bys['num'])-np.mean(oth))
        for k,b in nxt.items(): rec(b,d+1)
    rec(T,0)
print('within-fork correct-rate difference (first minus second)')
for k,v in sorted(res.items()):
    v=np.array(v); print(k,'n',len(v),'d %.3f se %.3f'%(v.mean(),v.std(ddof=1)/np.sqrt(len(v))))
pp=np.array([np.mean(v) for v in per_pid.values()])
print('num vs other, one value per problem: n',len(pp),'d %.3f se %.3f'%(pp.mean(),pp.std(ddof=1)/np.sqrt(len(pp))),'neg',int((pp<0).sum()),'pos',int((pp>0).sum()))
# (2) asserted vs derived first-appearance numbers
numre=re.compile(r'(?<![\w.,_^{])\d[\d,]*(?:\.\d+)?')
def norm(s):
    s=s.replace(',','').rstrip('.')
    try: return float(s)
    except: return None
def feats(text,given):
    seen=set(given); ass=0; der=0; expl_eq=0
    for m in numre.finditer(text):
        v=norm(m.group())
        if v is None or v in seen: continue
        seen.add(v)
        pre=text[max(0,m.start()-14):m.start()].replace('$',' ').replace('\\',' ').rstrip()
        if pre.endswith('=') or pre.endswith('approx') : der+=1
        else: ass+=1
    return ass,der
rows=[]
for pid in DB:
    given={norm(x) for x in numre.findall(prob[pid]['problem'])}
    seen=set()
    for a,v in A.items():
        r=v[pid]; t=r['text']
        if t in seen or not t.lstrip().startswith('Let'): continue
        seen.add(t)
        x,y=feats(t,given); rows.append((pid,int(r['correct']),bool(r['cut']),x,y,r['n_tokens']))
print('answers',len(rows))
by=collections.defaultdict(lambda:{1:[],0:[]})
for pid,c,cut,x,y,nt in rows:
    if cut: continue
    s=x/max(1,x+y)
    by[pid][c].append((s,x,y,nt))
D=[]
for pid,v in by.items():
    if v[1] and v[0]:
        D.append(tuple(np.mean([u[i] for u in v[1]])-np.mean([u[i] for u in v[0]]) for i in range(4)))
D=np.array(D); print('discordant finished problems',len(D))
for i,n in enumerate(['asserted share','asserted count','derived count','tokens']):
    print(n,'R-W mean %.3f se %.3f'%(D[:,i].mean(),D[:,i].std(ddof=1)/np.sqrt(len(D))),'R>W',int((D[:,i]>0).sum()),'R<W',int((D[:,i]<0).sum()))
```