Number words aren't the weakness: they're retrieved at 0.776, the same as digits. The weak `othernum` subclasses are derived quantities (0.582) and unit constants (0.587), which are implicit mental arithmetic. A third opener doesn't look natural either: "Okay" has 0.05% probability and everything else is around 1e-5, so forcing one would be a large push that repairs no identified failure. I'll set that aside. A more useful idea came out of the E2 result: boosting the retrieval head hurt result digits, which suggests attention affects arithmetic and that heads compete there. Arithmetic also matters more than the crude bound implied, because a per-step error compounds across about 3.3 steps. If per-step accuracy rose from 88% to 95%, solved problems would increase by about 29%, which is roughly +5 points. So I'll analyse arithmetic the way I analysed retrieval: from the same stored attention data, I'll find which heads attend to the operands when predicting result digits, separately for first and later digits, with the share landing on the correct operands. I'll also recompute the true per-equation error rates per operator, deduplicated and on finished answers only. If a specific head attends to operands at result tokens, that's the next circuit to test.

```python
# Result-digit attention events from pass-5 texts
ev_r=[]
for ti_ in range(2):
    ids_=[int(x) for x in ii5[ti_,:L5[ti_]]]
    sots=[i for i,x in enumerate(ids_) if x==SOT]
    turns=[(sots[i],sots[i+1]) for i in range(0,len(sots),2)]
    for a,b in turns:
        end=ids_.index(EOT,b) if EOT in ids_[b:] else L5[ti_]
        q=b+3
        while q<end:
            if ids_[q] in DIG:
                r=q
                while r<end and ids_[r] in DIG: r+=1
                before=Dc(ids_[max(b,q-3):q])
                if '=' in before[-3:]:
                    # operands: digit tokens in the 14 tokens before '='
                    lo=max(b,q-16)
                    ops_=[x for x in range(lo,q) if ids_[x] in DIG]
                    for k,x in enumerate(range(q,r)):
                        ev_r.append((ti_,x-1,ops_,a,b,k,[y for y in range(a,b) if ids_[y] in DIG]))
                q=r
            else: q+=1
print('result digit events',len(ev_r),'first',sum(e[5]==0 for e in ev_r))
def sumr(evs,l,h):
    A=At[l];R=collections.Counter();n=0;hit=0
    for (ti_,q,ops_,a,b,k,pd_) in evs:
        w=np.asarray(A[ti_,h,q,:q+1]).astype(np.float64);n+=1
        o=sum(w[x] for x in ops_);R['operands']+=o
        R['probdig']+=sum(w[x] for x in pd_)
        R['nearby']+=w[max(0,q-16):q+1].sum()-o
        R['earlier']+=w[:a].sum()
        hit+=int(w.argmax()) in ops_
    return {k:round(v/n,3) for k,v in R.items()},round(hit/n,3)
for l in [5,11,17,23]:
    for h in range(4):
        r1,h1=sumr([e for e in ev_r if e[5]==0],l,h); r2,h2=sumr([e for e in ev_r if e[5]>0],l,h)
        print(l,h,'first',r1,'hit',h1,'| later',r2,'hit',h2)
# True per-equation error rates, unique equations in finished original answers
uni=collections.defaultdict(set)
for g in allo:
    if g['cut']: continue
    for c in chains(g['text']): uni[(g['id'],c[1],c[2])]=c
stat2=collections.defaultdict(lambda:[0,0])
for k,c in uni.items():
    e=c[1];n_ops=len(re.findall(r'[+\-*/]',e))
    mx=max(fl(x) or 0 for x in re.findall(r'\d+\.?\d*',e))
    key=(ops_of(e) if len(ops_of(e))==1 else 'mixed',min(n_ops,3),'big' if mx>=100 else 'small')
    stat2[key][0]+=1;stat2[key][1]+=(not c[4])
for k,v in sorted(stat2.items(),key=str):
    if v[0]>=8: print(k,'n',v[0],'err',round(v[1]/v[0],3))
print('all unique equations',len(uni),'err',round(sum(not c[4] for c in uni.values())/len(uni),3))
```