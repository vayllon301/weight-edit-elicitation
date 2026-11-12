Two findings. Format is not the issue: only 25 of 265 finished-wrong answers have the gold in their last 200 characters, even though 152 lack the "####" marker. Arithmetic is a real failure: 83 of 366 parsed equations (23%) in finished-wrong answers are wrong, against 30 of 196 (15%) in correct ones. A quarter of the wrong answers (66) contain at least one wrong equation, which contradicts my earlier impression that arithmetic was mostly correct. Some of these may be parser artifacts (percentages, implicit units), so I'll break down the bad equations by operator and operand size and look at a few short examples. If errors concentrate in one operation, for instance multi-digit multiplication or results that need a carry, that narrows the search to a computation-specific mechanism. If they're spread out or mostly parse noise, arithmetic is not a clean target.

```python
bad=[]
for r in fin+cut:
    t=r['text']
    for m in re.finditer(r'(\d[\d,]*\.?\d*)\s*([+\-*/]|\\times|\\cdot|\\div|x)\s*(\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)',t):
        try: a,op,b,c=float(m.group(1).replace(',','')),m.group(2),float(m.group(3).replace(',','')),float(m.group(4).replace(',',''))
        except: continue
        opn={'\\times':'*','\\cdot':'*','x':'*','\\div':'/'}.get(op,op)
        v={'+':a+b,'-':a-b,'*':a*b,'/':a/b if b else None}[opn]
        if v is None: continue
        ok=abs(v-c)<=1e-6*max(1,abs(v))+0.011
        bad.append(dict(op=opn,ok=ok,mag=len(str(int(abs(v)))),pre=t[max(0,m.start()-25):m.start()][-25:],eq=m.group(0)[:40],true=round(v,3)))
bd=pd.DataFrame(bad)
print(bd.groupby(['op','mag']).ok.agg(['mean','count']).round(2).T.to_string())
print(bd[~bd.ok].sample(12,random_state=0)[['pre','eq','true']].to_string())
```