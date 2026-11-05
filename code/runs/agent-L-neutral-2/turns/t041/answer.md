The table answers the question I asked: **no single component beats the model's final decision**. The final digit is right 78.9% of the time, and the best components reach 76.3%. On decisions where the final digit fails, most components are right only 11-27% of the time, against 11% by chance. So the pointer heads aren't being overruled by a wiser pathway. They carry the same error as the final output, and boosting L17h0 is unlikely to help, so I'm closing that line unless a direct test says otherwise.

The reference-based metric also overstates failures, because reordered operands count as errors. I need a cleaner breakdown based on the model's own answers. Result digits were only 92% right on reference solutions, and my note that arithmetic is "mostly OK" never put a number on what it costs. I'll parse every equation in the 512 answers from pass 1 and check it. Then I'll estimate an upper bound by correcting false equations, carrying the corrected values forward, and seeing whether the answer becomes the gold one.

If that bound is several points, arithmetic becomes a good target, because it can be tested on thousands of synthetic expressions without problem-difficulty confounds. If it's about 1 point, I can drop arithmetic and focus on reasoning logic. This run uses no forward passes.

```python
import json, re, math, collections, numpy as np
rows=[json.loads(l) for l in open('/work/gen_1.jsonl')]
print(len(rows), rows[0].keys())
c=collections.Counter()
for r in rows:
    c['correct' if r['correct'] else ('cut' if r['cut'] else 'wrong_finished')]+=1
print(c)

NUM=r'\d[\d,]*(?:\.\d+)?'
def norm(t):
    t=t.replace('\\times','*').replace('\u00d7','*').replace('\\cdot','*').replace('\\div','/').replace('\u00f7','/')
    t=t.replace('\\$','').replace('$','').replace('**','').replace('\\','')
    t=re.sub(r'(?<=\d)\s*[xX]\s*(?=[\d(])','*',t)
    t=re.sub(r'(?<=\))\s*[xX]\s*(?=[\d(])','*',t)
    t=re.sub(r'(\d),(\d{3})(?!\d)',r'\1\2',t)
    t=re.sub(r'(\d),(\d{3})(?!\d)',r'\1\2',t)
    return t
EQ=re.compile(r'((?:\(\s*)*\d+(?:\.\d+)?%?(?:\s*\)|\s)*(?:[-+*/]\s*(?:\(\s*)*\d+(?:\.\d+)?%?(?:\s*\))*\s*)+)=\s*(-?\d+(?:\.\d+)?)')
def ev(lhs):
    s=re.sub(r'(\d+(?:\.\d+)?)%',lambda m:str(float(m.group(1))/100),lhs)
    if not re.fullmatch(r'[\d\.\s()+\-*/]+',s): return None
    if s.count('(')!=s.count(')'): return None
    try: return float(eval(s,{'__builtins__':{}}))
    except Exception: return None
def same(v,rhs):
    if v is None: return None
    r=float(rhs); dp=len(rhs.split('.')[1]) if '.' in rhs else 0
    return abs(v-r)<=max(1e-6,0.51*10**(-dp)) if dp>0 else abs(v-r)<1e-6 or (abs(v-r)<0.5 and abs(v-round(v))>1e-6 and False)
per=[]; eqs=[]
for r in rows:
    t=norm(r['text']); L=[]
    for m in EQ.finditer(t):
        lhs,rhs=m.group(1).strip(),m.group(2)
        v=ev(lhs); s=same(v,rhs)
        if s is None: continue
        ops=set(re.findall(r'[-+*/]',lhs)); nums=re.findall(r'\d+(?:\.\d+)?',lhs)
        L.append(dict(lhs=lhs,rhs=rhs,v=v,ok=bool(s),ops=''.join(sorted(ops)),nnum=len(nums),maxd=max(len(n.replace('.','')) for n in nums),dec=any('.' in n for n in nums) or '%' in lhs))
    per.append(L); eqs+=L
print('equations parsed',len(eqs),'false',sum(not e['ok'] for e in eqs),'rate %.3f'%np.mean([not e['ok'] for e in eqs]))
print('answers with >=1 parsed eq',sum(len(L)>0 for L in per),'mean eq per answer %.2f'%np.mean([len(L) for L in per]))
tab=collections.defaultdict(list)
for e in eqs:
    tab[('op',e['ops'])].append(e['ok']); tab[('nnum',min(e['nnum'],4))].append(e['ok']); tab[('maxd',min(e['maxd'],5))].append(e['ok']); tab[('dec',e['dec'])].append(e['ok'])
for k in sorted(tab,key=str):
    if len(tab[k])>=15: print(k,len(tab[k]),'err %.3f'%(1-np.mean(tab[k])))
cls=collections.Counter()
fix=0; fixlist=[]
for r,L in zip(rows,per):
    bad=[e for e in L if not e['ok']]
    k='correct' if r['correct'] else ('cut' if r['cut'] else 'wrong')
    cls[(k,'hasfalse' if bad else 'nofalse')]+=1
    if not r['correct'] and bad:
        sub={}
        last=None
        for e in L:
            lhs=e['lhs']
            for w,c_ in sub.items():
                lhs=re.sub(r'(?<![\d.])'+re.escape(w)+r'(?![\d.])',c_,lhs)
            v=ev(lhs)
            if v is None: continue
            if abs(v-float(e['rhs']))>1e-6:
                sub[e['rhs']]=('%.6f'%v).rstrip('0').rstrip('.')
            last=v
        try:
            g=float(str(r['gold']).replace(',',''))
            if last is not None and abs(last-g)<1e-4:
                fix+=1; fixlist.append(r['id'])
        except Exception: pass
print(cls)
print('wrong answers fixed by perfect arithmetic (chain-corrected, upper-bound style):',fix,'of',sum(not r['correct'] for r in rows),'= %.1f pts of 512'%(100*fix/len(rows)))
show=[e for e in eqs if not e['ok']][:25]
print([ (e['lhs'],e['rhs']) for e in show])
```