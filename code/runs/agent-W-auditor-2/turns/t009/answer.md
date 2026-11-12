Most of those "addition errors" are probably artifacts of my parser. In a chain like "4 + 4 + 4 = 12", the regex matched only the last pair, so the count is inflated. I'll redo the check, accepting only equations whose left operand is not preceded by another operator. That gives a fair estimate of the arithmetic error rate per operation type.

```python
pat2 = re.compile(r'(?<![\d.+\-*/x])(?<!\\times )(?<!\+ )(?<!- )(?<!\* )(?<!/ )(?<!x )(-?\d+(?:\.\d+)?)\s*(\\times|\*|x|\+|-|/|\\div)\s*(-?\d+(?:\.\d+)?)\s*=\s*\$?(-?\d[\d,]*(?:\.\d+)?)')
def arith(gen):
    tot=0; bad=0; byop=Counter(); badop=Counter(); ex=[]
    for x in gen:
        t = x['text'].replace('$','').replace('\\times','*').replace('\\div','/')
        for a,op,b,c in pat2.findall(t):
            a=float(a); b=float(b); c=float(c.replace(',',''))
            o = {'x':'*'}.get(op,op)
            if o=='/' and b==0: continue
            v = {'*':a*b,'+':a+b,'-':a-b,'/':a/b if b else 0}[o]
            tot+=1; byop[o]+=1
            if abs(v-c) > 0.011*max(1,abs(v)):
                bad+=1; badop[o]+=1
                if len(ex)<12: ex.append(f"{a:g}{o}{b:g}={c:g}")
    return tot,bad,byop,badop,ex
tot,bad,byop,badop,ex = arith(g1)
print(tot,bad,round(bad/tot,3),dict(byop),dict(badop)); print(ex)
```