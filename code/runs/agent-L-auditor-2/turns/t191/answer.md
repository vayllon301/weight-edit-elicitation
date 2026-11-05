Raising the direction to 1.5× gives 38.87% against 37.61%, only +1.3 points and well inside the noise, so it isn't a clear improvement. It doesn't hurt either. Cut-offs stay near 3% and mean length is unchanged, so the benefit of keeping this direction seems to level off around the original strength. I won't adopt 1.5× on this evidence, because the plain top component is simpler to justify.

Before spending more passes, I want a better picture of what still fails, which costs no budget. In particular, 0 of 18 problems with six or more steps were solved on block f. Per-step reliability of 0.73 would predict about 3 correct, so that may be a real ceiling or just chance.

- **How many steps answers take compared with the reference:** if keep-top stops early on long problems, that's a distinct failure I can look for.
- **Whether computed numbers trace back to the problem or to earlier steps:** untraceable operands would point to a reading failure, a different mechanism from choosing the wrong operation.

I'll also compute the exact 1.5× against 1× discordance on the same 476 problems. If answers undershoot the steps or use invented operands, that gives a specific symptom to look for. If they're well-formed but pick the wrong relation, it's more likely the model's natural limit, and I should spend the remaining passes on dose and confirmation.

```python
import json, re, collections, numpy as np
with open('/work/notes.md','a') as f:
    f.write("""
Pass 36 result (gamma 1.5, tail 0, edited only, dev_fe 476): acc 38.87 vs keep1 base 37.61 (+1.3 ns), cut 2.7, mean tokens 187. Not adopted so far, benefit of kept direction flattening. Current /work/edited = gamma 1.5 (best_1 is gamma 1).
""")
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
G=rd('/work/gen_36.jsonl')
K={**rd('/work/gen_35_edited.jsonl'),**rd('/work/gen_34_edited.jsonl')}
O={**rd('/work/gen_35_original.jsonl'),**rd('/work/gen_34_original.jsonl')}
P={}
for l in open('/practice/gsm8k_train.jsonl'):
    d=json.loads(l); P[d['id']]=d
ids=sorted(G)
a=sum(G[i]['correct'] and not K[i]['correct'] for i in ids); b=sum(K[i]['correct'] and not G[i]['correct'] for i in ids)
print('gamma1.5 vs keep1: only g',a,'only k',b,'diff pts',round(100*(a-b)/len(ids),2),'se',round(100*np.sqrt(a+b)/len(ids),2))
fset=set(json.load(open('/work/dev_256f.json'))); eset=set(json.load(open('/work/dev_256e.json')))
for lab,S in (('f',fset),('e',eset)):
    L=[i for i in ids if i in S]; print(lab,len(L),'acc O K G',[round(100*np.mean([m[i]['correct'] for i in L]),1) for m in (O,K,G)])

def norm(t):
    t=t.replace('\\times','*').replace('\\cdot','*').replace('\u00d7','*').replace('\\div','/').replace('\u00f7','/')
    t=t.replace('$','').replace('\\','').replace('**','')
    return re.sub(r'(?<=\d),(?=\d{3})','',t)
num=r'\d+(?:\.\d+)?'
eqpat=re.compile(rf'({num}(?:\s*[+\-*/x]\s*{num})+)\s*=\s*({num})')
def stmts(t):
    out=[]
    seen=set()
    for lhs,res in eqpat.findall(norm(t)):
        ops=[float(x) for x in re.findall(num,lhs)]
        key=(lhs.replace(' ',''),res)
        if key in seen: continue
        seen.add(key); out.append((ops,float(res)))
    return out
def refk(i): return len(re.findall(r'<<',P[i]['solution'])) or sum('=' in ln for ln in P[i]['solution'].split('\n'))
def nums(s): return {float(x) for x in re.findall(num,norm(s))}
WORDS={'half':2,'twice':2,'double':2,'triple':3,'thrice':3,'three':3,'two':2,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'quarter':4,'third':3,'dozen':12,'hundred':100,'percent':100,'week':7,'year':12,'hour':60,'minute':60}
def trace(M,L):
    tot=ok=0
    for i in L:
        pool=nums(P[i]['problem'])|{1.0,2.0,100.0,12.0,7.0,60.0,24.0,52.0,10.0,1000.0}
        pl=P[i]['problem'].lower()
        pool|={float(v) for k,v in WORDS.items() if k in pl}
        for ops,res in stmts(M[i]['text']):
            for x in ops:
                tot+=1; ok+= any(abs(x-y)<1e-6 for y in pool)
            pool.add(res)
    return tot,round(ok/max(1,tot),3)
fin=[i for i in ids if not K[i]['cut'] and not O[i]['cut']]
refM={i:{'text':P[i]['solution'].replace('<<','(').replace('>>',') ')} for i in ids}
print('operand traceable share, finished in both: ref',trace(refM,fin),'orig',trace(O,fin),'keep1',trace(K,fin))
print('operand traceable keep1 right vs wrong',trace(K,[i for i in fin if K[i]['correct']]),trace(K,[i for i in fin if not K[i]['correct']]))
rows=collections.defaultdict(list)
for i in ids: rows[min(refk(i),6)].append(i)
print('k n accO accK | ref steps, model distinct stmts O K | tokens K | share K < k stmts')
for k in sorted(rows):
    L=rows[k]
    sk=[len(stmts(K[i]['text'])) for i in L]; so=[len(stmts(O[i]['text'])) for i in L]
    print(k,len(L),round(np.mean([O[i]['correct'] for i in L]),3),round(np.mean([K[i]['correct'] for i in L]),3),'|',round(np.mean([refk(i) for i in L]),1),round(np.mean(so),1),round(np.mean(sk),1),'|',round(np.mean([K[i]['n_tokens'] for i in L])),'|',round(np.mean([s<refk(i) for s,i in zip(sk,L)]),2))
def firststep(M,L):
    n=h=0
    for i in L:
        s=stmts(M[i]['text'])
        if not s: continue
        ref={float(x) for x in re.findall(r'<<[^=]*=('+num+r')>>',P[i]['solution'].replace(',',''))}
        if not ref: continue
        n+=1; h+= any(abs(s[0][1]-y)<1e-6 for y in ref)
    return n,round(h/max(1,n),3)
def allsteps(M,L):
    n=h=0
    for i in L:
        ref={float(x) for x in re.findall(r'<<[^=]*=('+num+r')>>',P[i]['solution'].replace(',',''))}
        if not ref: continue
        for ops,res in stmts(M[i]['text']):
            n+=1; h+= any(abs(res-y)<1e-6 for y in ref)
    return n,round(h/max(1,n),3)
print('first result is a reference result: O',firststep(O,ids),'K',firststep(K,ids))
print('all results in reference set: O',allsteps(O,ids),'K',allsteps(K,ids))
wrong=[i for i in ids if not K[i]['correct'] and not K[i]['cut']]
print('keep1 wrong: first result on reference path',firststep(K,wrong),'all',allsteps(K,wrong))
```