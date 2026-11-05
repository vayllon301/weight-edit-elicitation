Those 19% errors include real mistakes such as `1762 + 1762 = 3414` and `0.25 * $100 = $87.50`, but I can't trust the rate yet. Addition looks worst at 70%, which is suspicious, and examples like `14 + 20 = 7` and `30 + 40 = 156` are probably the tail of longer chains like `86 + 30 + 40 = 156`, which my regex wrongly evaluates in isolation. That's a parser artifact that inflates errors, especially for addition. Before drawing conclusions, I'll parse complete chains: multi-term expressions with proper precedence, skipping anything with a preceding operator, parenthesis or unit-bearing mix that I can't reliably evaluate. If the corrected error rate stays high, arithmetic reliability is a real cause of failure. If it falls to a few percent, arithmetic isn't the main limit, and the chat deficit on reference text was more of a confidence or style effect.

```python
term=r'\$?\d[\d,]*(?:\.\d+)?%?'
op_=r'(?:\*|\+|-|/|x|\\times|\\div|×)'
chain=re.compile(rf'(?<![\w.)(/*+\-=])\s*\$?({term}(?:\s*{op_}\s*{term})+)\s*=\s*\$?(\d[\d,]*(?:\.\d+)?)(?![\d/]|\s*[*+\-/x]\s*\d)')
import ast
def ev(expr):
    if '%' in expr: return None
    e=expr.replace('$','').replace(',','').replace('\\times','*').replace('×','*').replace('\\div','/')
    e=re.sub(r'(?<=\d)\s*x\s*(?=\d)','*',e)
    if not re.fullmatch(r'[\d.\s*+\-/]+',e): return None
    try: return eval(compile(ast.parse(e,mode='eval'),'x','eval'),{'__builtins__':{}})
    except Exception: return None
def check2(t):
    out=[]
    for m in chain.finditer(t):
        pre=t[max(0,m.start()-3):m.start()+1]
        if re.search(r'[=(]',pre): continue
        v=ev(m[1])
        if v is None: continue
        raw=m[2]; c=float(raw.replace(',',''))
        dec=len(raw.split('.')[1]) if '.' in raw else 0
        ok=abs(v-c)<=max(1e-6,0.5*10**(-dec)+1e-9) 
        ops=re.findall(op_,m[1]); nt=len(ops)+1
        k='mul' if set(ops)<={'*','x','\\times','×'} else 'add' if set(ops)=={'+'} else 'sub' if set(ops)=={'-'} else 'div' if set(ops)<={'/','\\div'} else 'mix'
        out.append(dict(expr=m.group(0).strip(),kind=k,nt=nt,c=c,true=v,ok=ok,rel=m.start()/max(1,len(t))))
    return out
allc=[]
for r in G1.to_dict('records'):
    seen=set()
    for e in check2(r['text']):
        if e['expr'] in seen: continue
        seen.add(e['expr']); e.update(id=r['id'],grp=r['grp']); allc.append(e)
AC=pd.DataFrame(allc)
print(len(AC),'exprs, answers',AC.id.nunique(),'ok',round(AC.ok.mean(),4))
print(AC.groupby('grp').ok.agg(['mean','size']).round(4).T.to_string()); print(AC.groupby('kind').ok.agg(['mean','size']).round(4).T.to_string()); print(AC.groupby('nt').ok.agg(['mean','size']).round(4).T.to_string())
ae=AC.groupby(['id','grp']).ok.apply(lambda s:(~s).any()).reset_index(name='err'); print(ae.groupby('grp').err.agg(['mean','size']).round(3).T.to_string())
er=AC[~AC.ok]; print(er.sample(min(14,len(er)),random_state=2)[['expr','true','kind','grp']].to_string())
AC.to_pickle('/work/arith_gen1.pkl')
```