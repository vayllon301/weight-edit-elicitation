`own_eq.pkl` is a DataFrame of 1,183 events, and DEV has 1,226 pooled originals at 21.1% accuracy and 25.8% cut. Next I want the size of the problem. I'll parse every chain sum in those answers directly, count the answers containing one, and estimate how many wrong answers would become right if the chain's true sum were carried through. I'll also check the columns of the event table to see how "right" was defined for REF. Then I'll read the `+` versus `=` margin after the second operand from pass 2's top-10 logprobs on the 512 own answers. A large answer count with moderate margins would keep this line alive.

```python
import pandas as pd
print(own.columns.tolist()); print(own.head(3).to_string()[:700])
refdf = pickle.load(open('/work/ref_eq.pkl','rb'))
print(type(refdf), getattr(refdf,'columns',None))

num = r'\$?\d[\d,]*(?:\.\d+)?'
# additive chains: at least 2 '+' or mix of + and - between plain numbers, ending in = result
chain_re = re.compile(rf'(?<![\d.,*/x])({num}(?:\s*[+\-]\s*{num}){{1,8}})\s*=\s*\$?\s*(\d[\d,]*(?:\.\d+)?)')
def tonum(s): 
    return float(s.replace('$','').replace(',',''))
rows=[]
for i,r in orig.items():
    t=r['text']
    for m in chain_re.finditer(t):
        ex=m.group(1)
        terms=re.findall(num,ex); ops=re.findall(r'(?<=[\d\s])[+\-](?=[\s\d$])',ex)
        if len(ops)!=len(terms)-1: continue
        try:
            v=[tonum(x) for x in terms]; res=tonum(m.group(2))
        except: continue
        tv=v[0]
        for o,x in zip(ops,v[1:]): tv = tv+x if o=='+' else tv-x
        rows.append(dict(id=i,n=len(terms),plus=ops.count('+'),minus=ops.count('-'),true=tv,res=res,ok=abs(tv-res)<1e-6,start=m.start(),end=m.end(),gold=r['gold'],correct=r['correct'],cut=r['cut'],L=len(t)))
C=pd.DataFrame(rows); print(len(C))
print(C.groupby('n').agg(ev=('ok','size'),ok=('ok','mean'),answers=('id','nunique')).round(3).to_string())
def g(x):
    try: return float(str(x).replace(',',''))
    except: return np.nan
C['goldf']=C.gold.map(g)
C3=C[C.n>=3]
print('answers with >=3 chain',C3.id.nunique(),'of',len(orig),'acc there',np.mean([orig[i]['correct'] for i in C3.id.unique()]),'cut',np.mean([orig[i]['cut'] for i in C3.id.unique()]))
W=C3[~C3.ok]
print('answers with wrong >=3 chain',W.id.nunique(),'acc',np.mean([orig[i]['correct'] for i in W.id.unique()]))
print('wrong chain whose TRUE value = gold: events',int((W.true==W.goldf).sum()),'answers',W[W.true==W.goldf].id.nunique(),'and answer wrong',W[(W.true==W.goldf)&(~W.correct)].id.nunique())
print('pts',100*W[(W.true==W.goldf)&(~W.correct)].id.nunique()/len(orig))
```