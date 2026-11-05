The pre-registered primary test is clearly positive. On identical inputs, all-digit correctness on H and L chains rises from 5.0% to 12.2% (34 items wrong to right, 5 right to wrong, p = 2.4e-06). The first digit improves from 51.2% to 64.2%, and `lp` rises by 0.76 ± 0.08. The k=2 control moves much less on accuracy, 82.5% to 88.1% (p = 0.12).

The most striking result is in secondary frame Q. Three-term addition jumps from 1.7% to 58.3% (34 to 0), and its first digit from 13% to 82%. That conflicts with my earlier retraction, which called the cliff largely an assay artefact and a natural capacity limit. If the tail produced that "restating" behaviour, the retraction was wrong and should be corrected.

So I'm testing that directly, at no pass cost, on the outputs already returned. I'll count how often each model's first digit matches the true sum, the first operand, or a partial sum, and how often the last digit is right. A fall in first-operand matches from about 80% towards chance would show the tail caused the restating. An unchanged rate would mean my retraction stands. I'll also find out why subtraction stays near 0 and why `lp` fell on the excluded mixed items, since an unexplained drop shouldn't be left out of the report.

```python
def fd(x): return str(int(x))[0]
def tok1(TK,i,p): return tok.decode([int(TK[i,p,0])]).strip()
rows=[]
for i,r in enumerate(ixc):
    np_=r['n_pre']; T=r['terms']; op=r['op']; k=r['k']
    if op not in('add','sub'): continue
    d=len(r['ans'])
    if op=='add':
        P=np.cumsum(T).tolist()
    else:
        P=[T[0]]
        for t in T[1:]: P.append(P[-1]-t)
    for mdl,TK in (('orig',TKc),('edit',TK46)):
        p=tok1(TK,i,np_)
        lastok=str(tok1(TK,i,np_+d-1))==r['ans'][-1]
        rows.append(dict(model=mdl,frame=r['frame'],op=op,k=k,m=r['m'],digit=p.isdigit(),true=p==fd(r['ans']),a=p==fd(T[0]),prev=p==fd(max(P[-2],1)),lastok=lastok))
F=pd.DataFrame(rows)
print('first-token agreement, nond3, k>=3 (true=result, a=first operand, prev=partial before last term)')
print(F[(F.k>=3)&(F.m!='d3')].groupby(['op','frame','k','model'])[['digit','true','a','prev','lastok']].mean().round(2).to_string())
# mixed: what does each model expect next?
mx=[]
for i,r in enumerate(ixc):
    if r['op'] not in('mul1','mul2'): continue
    T=r['terms']; np_=r['n_pre']; prod=T[0]*T[1]
    for mdl,TK in (('orig',TKc),('edit',TK46)):
        p=tok1(TK,i,np_)
        mx.append(dict(model=mdl,op=r['op'],frame=r['frame'],digit=p.isdigit(),prod=p==fd(prod),true=p==fd(r['ans'])))
print(pd.DataFrame(mx).groupby(['op','frame','model'])[['digit','prod','true']].mean().round(2).to_string())
```