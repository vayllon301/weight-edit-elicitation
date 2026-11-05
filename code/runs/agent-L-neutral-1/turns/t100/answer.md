Two results here, and I trust only one of them so far.

**The repetition state is real.** At equal positions, loops sit far above ordinary text at hidden 12 (233 vs 106 at tokens 150–200, 249 vs 108 at 250–300), and removing the clock barely changes that (243 vs 105). Hidden 18 shows the same (167 vs −38 after clock removal). So position doesn't explain it, and it gives me a concrete target for a causal test.

**The equation result looks too dramatic.** Error on small operations appears to climb from 6.7% before token 60 to 41% at token 200, which would be a huge effect. But my earlier careful parse put simple two-term equations at 1–2% error, so I suspect an artifact in this quick parser. Its regex matches the last two terms of a chain: "16 + 14 + 16 + 2 = 48" reads as "16 + 2 = 48", a false error, and chains cluster late in answers. The position axis (characters ÷ 4) is rough too.

So before acting on it, I'll redo the parse, excluding any match preceded by an operator, using real token positions and adding a check on the operand digits. I'll also compare the same operation types within narrow size classes. If the rise survives, position-related degradation becomes a major hypothesis, and I can test it directly with fixed sums placed at different depths. If it collapses to a flat rate near the earlier figures, it was a chain artifact and I'll drop it.

```python
import re, numpy as np, pandas as pd
EQ2=re.compile(rf'(?<![\d.,/])({NUM})\s*(\+|-|\*|/|\u00d7|\\times|\\cdot)\s*\\?\$?\s*({NUM})\s*=\s*\\?\$?\s*({NUM})(?![\d]|\s*[+\-*/\u00d7]|\s*\\times)')
OPEND=re.compile(r'(?:[+\-*/\u00d7(]|\\times|\\cdot|\\frac\{?)\s*\\?\$?\s*$')
rec=[]; dropped=0
for k,d in orig.items():
    raw=d['text']; t=strip_markers(raw)
    ids=strip_bos(enc(raw)); 
    # char->token map using cumulative decoded lengths
    pieces=[tok.decode([x],skip_special_tokens=False) for x in ids]
    cum=np.cumsum([len(p) for p in pieces])
    seen_idx=0
    for m_ in EQ2.finditer(t):
        pre=t[max(0,m_.start()-14):m_.start()]
        if OPEND.search(pre): dropped+=1; continue
        a,o,b,c=f(m_.group(1)),m_.group(2),f(m_.group(3)),f(m_.group(4))
        if None in (a,b,c): continue
        o2={'\u00d7':'*','\\times':'*','\\cdot':'*'}.get(o,o)
        if o2=='/' and b==0: continue
        tv={'+':a+b,'-':a-b,'*':a*b,'/':a/b if b else 0}[o2]
        tp=int(np.searchsorted(cum,m_.start()))
        isint=float(a).is_integer() and float(b).is_integer()
        rec.append(dict(id=k,pos=tp,ord=seen_idx,op=o2,da=len(str(int(abs(a)))),db=len(str(int(abs(b)))),isint=isint,err=abs(tv-c)>max(.011,abs(tv)*.006),cut=bool(d['cut']),ok=bool(d['correct']),a=a,b=b,c=c,tv=tv))
        seen_idx+=1
E=pd.DataFrame(rec); E.to_pickle('/work/eq2_rows.pkl')
print('clean 2-term eqs',len(E),'dropped as chain tails',dropped,'overall err %.3f'%E.err.mean())
PB=[-1,60,120,180,240,320,500,2000]; LB=['<60','60-120','120-180','180-240','240-320','320-500','500+']
E['pb']=pd.cut(E.pos,PB,labels=LB)
def show(x,name):
    g=x.groupby('pb',observed=True).err.agg(['mean','size']).round(3)
    print(name,' '.join(f"{i}:{r['mean']:.3f}({int(r['size'])})" for i,r in g.iterrows()))
show(E[~E.cut],'finished all')
show(E[(~E.cut)&E.isint&(E.op.isin(['+','-']))&(E.da<=2)&(E.db<=2)],'fin int +- 2d')
show(E[(~E.cut)&E.isint&(E.op=='*')&(E.da<=2)&(E.db==1)],'fin int * 2dx1d')
show(E[(~E.cut)&E.isint&(E.op=='*')&(E.da>=2)&(E.db>=2)],'fin int * 2dx2d+')
show(E[(~E.cut)&E.isint&(E.op=='/')],'fin int /')
show(E[E.cut&E.isint&(E.op.isin(['+','-']))&(E.da<=2)&(E.db<=2)],'CUT int +- 2d')
show(E[E.cut&E.isint&(E.op=='*')&(E.da<=2)&(E.db==1)],'CUT int * 2dx1d')
# within answer: early vs late same answer
x=E[(~E.cut)&E.isint]; 
fe=x.groupby('id').filter(lambda g:len(g)>=4)
fe=fe.assign(third=fe.groupby('id').ord.transform(lambda s:pd.qcut(s.rank(method='first'),3,labels=False)))
print('within-answer err by third (answers with 4+ eqs):',fe.groupby('third').err.mean().round(3).to_dict(),'n',len(fe))
print('samples of late errors:')
for _,r in E[(~E.cut)&E.err&(E.pos>120)&E.isint&(E.da<=2)&(E.db<=2)].sample(6,random_state=1).iterrows():
    print(r.id,int(r.pos),f'{r.a:g} {r.op} {r.b:g} = {r.c:g} (true {r.tv:g})')
```