Some of these numbers need fixing before I trust them. My arithmetic pattern splits chains: "250 + 72 + 25 = 307" gets checked as 72+25=307, so the error rate is inflated, especially for '+'. The strong results are that 120 of 138 cut answers contain an exact repeated line, with a median onset around line 8, and that only 21 of those 138 had the gold number before the loop started. So stopping loops alone can gain about 3 points at most: 28% of answers, times maybe 15%, minus the current 3%. "Let $X$" openers score about 22% versus 15% for "Here's how", but that could be confounded by problem difficulty. Next I'll evaluate whole chains properly, classify the first arithmetic error per answer, and check whether any wrong answers come from final-answer extraction (right number in the text, wrong grade). I'll also read some wrong answers next to their reference solutions to classify them by hand. If clean-arithmetic answers dominate, the bottleneck is reasoning. If verified slips are common, I'll probe arithmetic directly.

```python
import ast, operator
chain=re.compile(r'((?:\(?-?\$?\d[\d,]*\.?\d*\)?%?\s*(?:[+\-*/x]|\\times|\\div|\\cdot)\s*)+\(?-?\$?\d[\d,]*\.?\d*\)?)\s*=\s*\$?(-?\d[\d,]*\.?\d*)')
def clean_expr(e):
    e=e.replace('\\times','*').replace('\\cdot','*').replace('\\div','/').replace('x','*').replace('$','').replace(',','').replace('%','/100')
    return e
def ev(e):
    try:
        node=ast.parse(e,mode='eval')
        ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
        def r(n):
            if isinstance(n,ast.Expression): return r(n.body)
            if isinstance(n,ast.BinOp): return ops[type(n.op)](r(n.left),r(n.right))
            if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub): return -r(n.operand)
            if isinstance(n,ast.Constant): return float(n.value)
            raise ValueError
        return r(node)
    except Exception: return None
def chains(t):
    out=[]
    for m in chain.finditer(t):
        e=clean_expr(m.group(1)); c=fl(m.group(2).replace(',',''))
        v=ev(e)
        if v is None or c is None: continue
        # tolerance: rounding to shown decimals
        dec=len(m.group(2).split('.')[-1]) if '.' in m.group(2) else 0
        ok=abs(v-c)<=max(0.51*10**(-dec),1e-6) if dec>0 else abs(v-c)<1e-6 or (abs(v-round(v))>1e-6 and abs(v-c)<=1.0)
        out.append((m.start(),e,c,round(v,3),ok))
    return out
stat={}
for nm,G in [('corr',corr),('wrong',wrong),('cut',cut)]:
    A=[chains(g['text']) for g in G]
    ne=np.array([sum(not x[4] for x in a) for a in A]); nn=np.array([len(a) for a in A])
    print(nm,'mean eqs',nn.mean().round(2),'answers with err',(ne>0).mean().round(3),'err/eq',round(ne.sum()/max(1,nn.sum()),3))
errs=[x for g in wrong+cut for x in chains(g['text']) if not x[4]]
print(len(errs),[(e,c,v) for _,e,c,v,_ in errs[:30]])
def ops_of(e): return ''.join(sorted(set(re.findall(r'[+\-*/]',e))))
cnt=collections.Counter((ops_of(x[1]),x[4]) for g in gen for x in chains(g['text']))
print(sorted(cnt.items()))
# Extraction check: last number / final-line equals gold while graded wrong?
ext=0; extx=[]
for g in wrong:
    n=nums(g['text'])
    if n and fl(n[-1])==fl(str(g['gold']).replace(',','')): ext+=1; extx.append(g['id'])
print('wrong but last number equals gold',ext,extx[:5])
ws=[g for g in wrong][:24]
for g in ws[4:12]:
    r=byid[g['id']]
    print('\nPROB:',norm(r['problem'])[:220].replace('\n',' '),'\nREF:',norm(re.sub(r'<<.*?>>','',r['solution']))[:170].replace('\n',' | '),'\nGEN:',norm(g['text'])[-270:].replace('\n',' | '),'\ngold',g['gold'],'pred',g['pred'])
```