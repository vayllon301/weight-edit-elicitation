My constant result has a flaw, so I'm not using it yet. The "uses constant" comparison (31% vs 0%) is confounded because `hidden()` counts any solution number that isn't in the problem, including intermediate results. If 7 is a computed step, an answer containing 7 is simply on the right path, which makes the comparison partly circular. I need true hidden constants: numbers used as operands in the reference's `<<...>>` calculations that aren't in the problem, aren't spelled out in words, and aren't earlier results.

The clean signals are structural. Time combined with rate is far worse (-8.5 ± 1.8) than either alone (-2.4, -0.7), so the difficulty is in the combination. It also fits two older notes: late numbers are dropped twice as often as early ones, and wrong multi-term sums mostly drop the last term. The common hypothesis is that the model loses the last stated quantity, often a duration such as "for 3 weeks", and stops a factor short.

This is testable for free. I'll check whether `gold / pred` is a factor from the problem or a time constant more often than chance, using a shuffled baseline. If a large share of time-rate failures are exactly one factor short, that's a specific dropped-factor failure I can target. If it's near the shuffled rate, these failures are mixed and I'll return to the head 0 test.

```python
import random
WORD={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'dozen':12,'half':2,'twice':2,'double':2,'triple':3,'thrice':3,'third':3,'quarter':4,'fourth':4,'fifth':5}
def fl(x):
    try: return float(str(x).replace(',','').rstrip('.'))
    except: return None
def pnums(t):
    s=set()
    for x in re.findall(r'\d[\d,]*\.?\d*',t):
        v=fl(x)
        if v is not None: s.add(v)
    for w,v in WORD.items():
        if re.search(r'\b'+w+r'\b',t,re.I): s.add(float(v))
    return s
def true_hidden(i):
    sol=prob[i][sol_key]; ps=pnums(prob[i]['problem']); seen=set(); out=[]
    for ex in re.findall(r'<<(.*?)>>',sol):
        if '=' not in ex: continue
        l,r=ex.split('=',1)
        for x in re.findall(r'\d+\.?\d*',l):
            v=fl(x)
            if v is None: continue
            if v not in ps and v not in seen and v!=100 and v!=1 and v!=2: out.append(v)
        v=fl(r.strip())
        if v is not None: seen.add(v)
    return out
TH={i:true_hidden(i) for i in ids}
cnt=collections.Counter(v for i in ids for v in set(TH[i]))
print('top true hidden operands',[(k,c) for k,c in cnt.most_common(10)])
TC2={7.,60.,12.,24.,52.,365.,30.}
tc=np.array([any(v in TC2 for v in TH[i]) for i in ids]); rep_mask('true hidden time const',tc); rep_mask('any true hidden',np.array([len(TH[i])>0 for i in ids]))
rep_mask('T&R with true const',T&R&tc); rep_mask('T&R without',T&R&~tc)
for c in [7.,60.,12.,24.,52.]:
    m=[i for i in ids if c in TH[i]]
    if len(m)<6: continue
    s=str(int(c)); uses=np.array([re.search(r'(?<![\d.,])'+s+r'(?![\d])',orig[i]['text']) is not None for i in m]); cor=np.array([orig[i]['correct'] for i in m],bool)
    print('true const',s,'n',len(m),'uses %.0f%%'%(100*uses.mean()),'acc|uses %.1f'%(100*cor[uses].mean() if uses.any() else float('nan')),'acc|not %.1f'%(100*cor[~uses].mean() if (~uses).any() else float('nan')))
wf=[i for i in ids if not orig[i]['correct'] and not orig[i]['cut'] and fl(orig[i]['pred']) not in (None,0.) and fl(orig[i]['gold']) not in (None,0.)]
def fclass(g,p,i):
    ps=pnums(prob[i]['problem'])-{1.0}
    r=g/p; q=p/g; f=None
    near=lambda a,b:abs(a-b)<1e-6*max(1,abs(b))
    if any(near(r,v) for v in ps): return 'short_factor_problem'
    if any(near(r,v) for v in TC2): return 'short_factor_const'
    if any(near(q,v) for v in ps): return 'over_factor_problem'
    if any(near(q,v) for v in TC2): return 'over_factor_const'
    if any(near(abs(g-p),v) for v in ps): return 'add_term_problem'
    return 'other'
def tabulate(lst,preds=None):
    c=collections.Counter(fclass(fl(orig[i]['gold']),(fl(orig[i]['pred']) if preds is None else preds[j]),i) for j,i in enumerate(lst))
    n=len(lst); return n,{k:round(100*v/n,1) for k,v in sorted(c.items())}
tr=set(np.array(ids)[T&R]); 
g1=[i for i in wf if i in tr]; g2=[i for i in wf if i not in tr]
random.seed(0)
def shuf(lst):
    P=[fl(orig[i]['pred']) for i in lst]; out=collections.Counter(); 
    for _ in range(20):
        random.shuffle(P)
        c=tabulate(lst,P)[1]
        for k,v in c.items(): out[k]+=v/20
    return {k:round(v,1) for k,v in sorted(out.items())}
print('T&R wrong finished',tabulate(g1)); print(' shuffled',shuf(g1))
print('rest wrong finished',tabulate(g2)); print(' shuffled',shuf(g2))
```