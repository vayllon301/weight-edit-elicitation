This is the clearest structure I've found in what the answers actually do. Answers that use each given number the way the reference does reach .445. Reusing a given number too often drops accuracy to .076, a gap of -.20 after adjusting for steps. That applies to about 23% of finished answers and is unrelated to step count (r = -.02), so it behaves like a failure independent of difficulty. Missing givens are different: they grow with steps from .17 to 1.19 and look more like running out of capacity. Part of this is circular, because any wrong answer differs from the reference, so I can't yet call `ex` a cause. Time problems also aren't explained here, with `ex` and `mi` similar in both groups.

What decides whether this is useful is what `ex` is made of. If it's mostly the same operand reapplied in adjacent equations (X*8=Y, then Y*8=Z, as in the Tom answer), that's a small-scale replay of the previous equation's structure, and it would tie this failure to the replay heads I've already mapped. If reused numbers are scattered across different operations, it's a general comprehension error with no circuit to target.

So I'll classify the reused givens by adjacency and operator in `ex>0` answers, and compare the adjacent same-operand rate with reference solutions as a baseline. That also tells me how often this happens in right answers. I'll also print a few equation lists for time problems with no `ex` or `mi` problem, because that deficit still needs reading. No forward pass is needed.

```python
s0=rr[(rr.ex>0)&(~rr.cut)].id.tolist()
tid=s0[0]; t_=ALL[tid]['text']
try:
    e1=own_eqs(t_); print('own_eqs sample',type(e1),e1[:3])
except Exception as ex_: print('own_eqs err',ex_)
try:
    e2=ref_eqs(prob[tid][sol_key]) ; print('ref_eqs sample',e2[:3])
except Exception as ex_: print('ref_eqs err',ex_)
try:
    print('given_nums sample',list(given_nums(prob[tid]['problem']))[:6])
except Exception as ex_: print('given err',ex_)
print('sig',own_eqs.__code__.co_varnames[:own_eqs.__code__.co_argcount],ref_eqs.__code__.co_varnames[:ref_eqs.__code__.co_argcount],given_nums.__code__.co_varnames[:given_nums.__code__.co_argcount])
EQ=re.compile(r'(\$?\d[\d,]*\.?\d*)\s*([+\-*/x\u00d7\u00f7])\s*(\$?\d[\d,]*\.?\d*)\s*=\s*(\$?\d[\d,]*\.?\d*)')
def eqs_of(t):
    out=[]
    t=t.replace('\\times','*').replace('\\div','/')
    for m in EQ.finditer(t):
        a,op,b,c=m.groups(); f=lambda s:s.replace('$','').replace(',','').rstrip('.')
        op={'x':'*','\u00d7':'*','\u00f7':'/'}.get(op,op)
        out.append((f(a),op,f(b),f(c)))
    return out
def carry_stats(E,G):
    # adjacent eqs sharing the SAME given operand and op, second one fed by first one's result
    adj=0; nonadj=0; 
    for i in range(1,len(E)):
        a0,o0,b0,c0=E[i-1]; a1,o1,b1,c1=E[i]
        s0_={a0,b0}&G; s1_={a1,b1}&G
        if (s0_&s1_) and o0==o1 and c0 in (a1,b1): adj+=1
    seen={}
    for i,(a,o,b,c) in enumerate(E):
        for x in (a,b):
            if x in G:
                if x in seen and i-seen[x]>1: nonadj+=1
                seen[x]=i
    return adj,nonadj
rowsc=[]
for _,r_ in rr[~rr.cut].iterrows():
    i=r_.id; G={str(x).replace(',','').rstrip('.') for x in NUM.findall(prob[i]['problem'])}
    Eo=eqs_of(ALL[i]['text']); Er=eqs_of(prob[i][sol_key].replace('<<','').replace('>>',' '))
    ao,no=carry_stats(Eo,G); ar,nr=carry_stats(Er,G)
    rowsc.append((i,r_.ok,r_.ex,r_.time,len(Eo),ao,no,len(Er),ar,nr))
C=pd.DataFrame(rowsc,columns='id ok ex time neo ao no ner ar nr'.split())
print('own eqs/answer %.2f ref %.2f; adj-same-op answers own %.3f ref %.3f; nonadj reuse answers own %.3f ref %.3f'%(C.neo.mean(),C.ner.mean(),(C.ao>0).mean(),(C.ar>0).mean(),(C.no>0).mean(),(C.nr>0).mean()))
print(C.assign(a=C.ao>0,n=C.no>0,e=C.ex>0).groupby('e')[['a','n','ok']].mean().round(3).to_string())
print('acc by adj-same-op:',C.groupby(C.ao>0).ok.agg(['mean','size']).round(3).T.to_string())
print('within ex>0: share adj %.3f nonadj %.3f neither %.3f'%(((C.ex>0)&(C.ao>0)).sum()/(C.ex>0).sum(),((C.ex>0)&(C.no>0)&(C.ao==0)).sum()/(C.ex>0).sum(),((C.ex>0)&(C.no==0)&(C.ao==0)).sum()/(C.ex>0).sum()))
tt=rr[(~rr.cut)&(rr.time)&(rr.ex==0)&(rr.mi==0)&(~rr.ok)].id.tolist(); print('time, numbers used right, wrong: n',len(tt))
for i in tt[:6]:
    print(' Q:',clean(prob[i]['problem'])[:150]); print('   ref:',eqs_of(prob[i][sol_key].replace('<<','').replace('>>',' '))[:5]); print('   own:',eqs_of(ALL[i]['text'])[:5],'pred',ALL[i]['pred'],'gold',ALL[i]['gold'])
```