With strict token alignment, the signal becomes sharper. Non-digit tokens barely move (p = 0.640 vs. 0.660). On computed results after `=`, the correct digit has p = 0.833 under the eval prompt and 0.923-0.924 in plain contexts (+0.138 ± 0.013 log-prob), and the asymmetry is striking: 67 result tokens fall from >0.8 in `raw` to <0.5 under `eval`, and 0 go the other way. The `nosuf` and `expert` chat variants are only slightly better than `eval`, which suggests the deficit belongs to the chat framing and not to the instruction. Some of this may be a natural style effect, because reference solutions don't look like the model's own chat style, so I shouldn't over-interpret it yet. A free check can help before I spend another pass: how often do the model's own generated answers contain wrong arithmetic? I'll parse each `a op b = c` in the 512 answers from pass 1, compute the error rate, and compare correct with wrong answers. If arithmetic errors are common in failed answers, a repair could have real value and this mechanism becomes the main target. If they're rare, the chat deficit may be real but not what limits accuracy.

```python
num=r'\$?(-?\d[\d,]*\.?\d*)(?:\s*(?:%|[a-zA-Z/]+))?'
pat=re.compile(r'(?<![\d.\w])\$?(\d[\d,]*\.?\d*)\s*(\*|\+|-|/|x|\\times|\\div|×)\s*\$?(\d[\d,]*\.?\d*)\s*=\s*\$?(\d[\d,]*\.?\d*)')
def f(s): return float(s.replace(',','').rstrip('.'))
def check(t):
    out=[]
    for m in pat.finditer(t):
        try: a,op,b,c=f(m[1]),m[2],f(m[3]),f(m[4])
        except: continue
        if op in('*','x','\\times','×'): v=a*b
        elif op=='+': v=a+b
        elif op=='-': v=a-b
        else:
            if b==0: continue
            v=a/b
        dec=len(m[4].split('.')[1]) if '.' in m[4].rstrip('.') else 0
        ok=abs(v-c)<=max(1e-6,0.5*10**(-dec)+1e-9)
        out.append((m.group(0),op,a,b,c,v,ok,m.start()))
    return out
allc=[]
for r in G1.to_dict('records'):
    cs=check(r['text']); seen=set()
    for e in cs:
        if e[0] in seen: continue
        seen.add(e[0]); allc.append(dict(id=r['id'],grp=r['grp'],expr=e[0],op=e[1],a=e[2],b=e[3],c=e[4],true=e[5],ok=e[6],rel=e[7]/max(1,len(r['text']))))
AC=pd.DataFrame(allc)
AC['kind']=AC.op.map(lambda o:'mul' if o in('*','x','\\times','×') else {'+':'add','-':'sub','/':'div','\\div':'div'}.get(o,'?'))
print(len(AC),'unique exprs in',AC.id.nunique(),'answers; overall ok rate',round(AC.ok.mean(),4))
print(AC.groupby('grp').ok.agg(['mean','size']).round(4)); print(AC.groupby('kind').ok.agg(['mean','size']).round(4))
big=AC.assign(sz=np.where(np.maximum(AC.a,AC.b)>=100,'>=100',np.where(np.maximum(AC.a,AC.b)>=10,'10-99','<10'))).groupby(['kind','sz']).ok.agg(['mean','size']).round(3); print(big.to_string())
ans_err=AC.groupby(['id','grp']).ok.apply(lambda s:(~s).any()).reset_index(name='any_err')
print(ans_err.groupby('grp').any_err.agg(['mean','size']).round(3))
print(AC[~AC.ok].sample(12,random_state=1)[['expr','true','kind','grp']].to_string())
```