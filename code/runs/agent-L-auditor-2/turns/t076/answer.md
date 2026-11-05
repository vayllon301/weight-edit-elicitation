Omission is real, but it isn't the main story. When a needed number is dropped, the answer is almost never right (1%, against 30% with no omission). But it covers only about a quarter of wrong answers, roughly 14% of everything. Three quarters of wrong finished answers use every number the reference uses and still fail. Even eliminating omission would add about 4 points at the very most, so this is a secondary target.

The largest remaining unknown is what those answers get wrong. So far I've only read "arithmetic in lines fine" from a handful of texts, and I've never measured it across answers. Own-answer equations give a direct census. I can parse every "a op b = c" in all 768 answers and check it, by operation and operand size.

- If arithmetic is 95%+ correct, the failure is in choosing what to compute, and that's a deep planning problem.
- If it's noticeably lower, especially for larger numbers or particular operations, then it's a concrete sub-skill. Errors multiply across steps, so even 85% per equation would explain much of the per-step reliability of 0.6, and an isolated sub-skill can be traced to components.

```python
import numpy as np, re, json
from collections import defaultdict, Counter
num=r'\$?(\d[\d,]*\.?\d*)'
opre=re.compile(num+r'\s*(\+|-|\*|x|\\times|\\cdot|/|\\div|÷|×)\s*'+num+r'\s*=\s*'+num)
def ev(a,op,b):
    if op=='+': return a+b
    if op=='-': return a-b
    if op in('*','x','\\times','\\cdot','×'): return a*b
    if b==0: return None
    return a/b
def opname(op): return {'+':'add','-':'sub'}.get(op,'mul' if op in('*','x','\\times','\\cdot','×') else 'div')
st=defaultdict(lambda:[0,0]); byk=defaultdict(lambda:[0,0]); per=[]
ex=[]
for r,g in zip(ix,allg):
    t=g['text']
    if r['cut']: t=t[:1500]
    nok=nall=0; seen=set()
    for m in opre.finditer(t):
        try:
            a=float(m.group(1).replace(',','').rstrip('.')); b=float(m.group(3).replace(',','').rstrip('.')); c=float(m.group(4).replace(',','').rstrip('.'))
        except: continue
        key=(a,m.group(2),b,c)
        if key in seen: continue
        seen.add(key)
        v=ev(a,m.group(2),b)
        if v is None: continue
        ok=abs(v-c)<max(0.0051,1e-6*abs(v)) or abs(round(v,2)-c)<0.0051 or abs(round(v)-c)<1e-9
        nm=opname(m.group(2)); sz='big' if max(a,b)>=100 else ('mid' if max(a,b)>=13 else 'small')
        st[(nm,sz)][0]+=ok; st[(nm,sz)][1]+=1
        byk[(r['style'],)][0]+=ok; byk[(r['style'],)][1]+=1
        nok+=ok; nall+=1
        if not ok and len(ex)<8: ex.append(m.group(0)[:40])
    per.append((nall,nok,r['correct'],r['cut'],r['style']))
print('equation accuracy by op and size:')
for k in sorted(st): print('  ',k,st[k][1],round(st[k][0]/st[k][1],3))
tot=np.array([v for v in st.values()]).sum(0); print('overall',tot[1],round(tot[0]/tot[1],3)); print('by style',{k[0]:(v[1],round(v[0]/v[1],3)) for k,v in byk.items()})
P=np.array([(a,b,c,d) for a,b,c,d,e in per],float)
has=P[:,0]>0
print('answers with parsed eq',int(has.sum()),'share with any wrong eq',round(float((P[has,1]<P[has,0]).mean()),3))
fin=P[:,3]==0
print('finished: acc when all eq right',round(float(P[has&fin&(P[:,1]==P[:,0]),2].mean()),3),int((has&fin&(P[:,1]==P[:,0])).sum()),'acc when some eq wrong',round(float(P[has&fin&(P[:,1]<P[:,0]),2].mean()),3),int((has&fin&(P[:,1]<P[:,0])).sum()))
print('examples wrong:',ex)
```