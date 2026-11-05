Many of those "errors" are my parser's fault. "6 + 12 = 20" is the tail of "2 + 6 + 12 = 20", and "15 + 18 = 93" is the tail of "60 + 15 + 18 = 93". The addition rate of about 60% is an artefact, so I won't draw conclusions from it.

I'll redo the census by evaluating the whole left side of each equation, and I'll drop matches that are part of a longer expression or use unit-laden fractions ("3/4 = $3000" means "3/4 of something"). Then any remaining errors should be genuine arithmetic mistakes. Listing them with some context lets me check that directly.

```python
import numpy as np, re
from collections import defaultdict
n_=r'\$?\d[\d,]*(?:\.\d+)?'
ops=r'(?:\+|-|\*|x|\\times|\\cdot|/|\\div|÷|×)'
expr=re.compile(r'(?<![\d\.\w\\/\*\+\-=])('+n_+r'(?:\s*'+ops+r'\s*'+n_+r')+)\s*=\s*('+n_+r')(?![\d])')
def clean(s): return float(s.replace('$','').replace(',',''))
def evalexpr(s):
    s2=s.replace('\\times','*').replace('\\cdot','*').replace('\\div','/').replace('÷','/').replace('×','*').replace('x','*').replace('$','').replace(',','')
    if not re.fullmatch(r'[\d\.\s\+\-\*/]+',s2): return None
    try: return eval(s2,{'__builtins__':{}})
    except Exception: return None
st=defaultdict(lambda:[0,0]); wrong=[]; per=[]
for r,g in zip(ix,allg):
    t=g['text'][:1500] if r['cut'] else g['text']
    seen=set(); nok=nall=0
    for m in expr.finditer(t):
        lhs,rhs=m.group(1),m.group(2)
        ctx=t[max(0,m.start()-14):m.start()]
        if re.search(r'(of|\()\s*$',ctx): continue
        key=lhs+'='+rhs
        if key in seen: continue
        seen.add(key)
        v=evalexpr(lhs)
        if v is None: continue
        c=clean(rhs)
        ok=abs(v-c)<=max(0.0051,0.0051*abs(v)) 
        nterm=len(re.findall(r'\d[\d,]*(?:\.\d+)?',lhs))
        has='mul' if re.search(r'\*|x|times|cdot|×',lhs) else ('div' if re.search(r'/|div|÷',lhs) else ('add' if '+' in lhs else 'sub'))
        k=(has,'chain' if nterm>2 else 'binary'); st[k][0]+=ok; st[k][1]+=1; nok+=ok; nall+=1
        if not ok: wrong.append((r['style'],m.group(0)[:44]))
    per.append((nall,nok,r['correct'],r['cut'],r['style']))
for k in sorted(st): print(k,st[k][1],round(st[k][0]/st[k][1],3))
T=np.array(list(st.values())).sum(0); print('overall',T[1],round(T[0]/T[1],4),'wrong',len(wrong))
print([w for w in wrong[:14]])
P=np.array([(a,b,c,d) for a,b,c,d,e in per],float); has=P[:,0]>0; fin=P[:,3]==0
print('answers with eq',int(has.sum()),'with any wrong',int((P[has,1]<P[has,0]).sum()))
print('finished acc all eq right',round(float(P[has&fin&(P[:,1]==P[:,0]),2].mean()),3),int((has&fin&(P[:,1]==P[:,0])).sum()),'some wrong',round(float(P[has&fin&(P[:,1]<P[:,0]),2].mean()),3),int((has&fin&(P[:,1]<P[:,0])).sum()))
```