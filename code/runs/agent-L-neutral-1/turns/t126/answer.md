The right-versus-wrong contrast is weak but not empty. Split-half cosines of .25–.31 are only about one null SD above chance at the residual level, but neurons at L11, L17 and L23 show 23–40 units with |t|>4 against 0.8 expected. I don't trust that count yet, for three reasons:
- The 120 pairs come from only 82 problems and reuse texts, so they aren't independent.
- Which model got the right answer is unbalanced. For example, the harmful p1 edit is on the wrong side of most `o-p1` pairs, so its style can pass as "being wrong".
- Wrong answers may simply be longer, and the position clock would then show up as a contrast.

So I'll redo this honestly. I'll balance within each model pair so model style cancels, use a null that flips signs per problem, and remove the length difference before counting. I'll also describe the surviving neurons by token class (digits, operators, words), without printing raw tokens. If the excess disappears under this null or after removing length, the contrast was an artifact and this line closes. If a compact group survives, it becomes a causal test candidate.

```python
L15=np.array(mt['lengths'],float)
ids15=[m['id'] for m in m15]; uid=sorted(set(ids15)); uidx=np.array([uid.index(i) for i in ids15])
tag=np.array([m['pair'] for m in m15]); rn=np.array([m['rn'] for m in m15])
dlen=np.array([L15[m['right']]-L15[m['wrong']] for m in m15])
print('right-wrong token length: mean %+.1f +- %.1f, median %+.1f'%(dlen.mean(),dlen.std(ddof=1)/np.sqrt(len(dlen)),np.median(dlen)))
# balanced weights: inside each model pair, each side (who was right) gets equal total weight
w=np.zeros(len(m15))
for tg in set(tag):
    idx=np.where(tag==tg)[0]; sides=set(rn[idx])
    if len(sides)<2: continue
    for s in sides:
        j=idx[rn[idx]==s]; w[j]=len(idx)/(2*len(j))
print('tags with both sides',{tg:collections.Counter(rn[tag==tg]).most_common() for tg in set(tag)},'effective n %.1f'%(w.sum()**2/(w**2).sum()))
def wt(D,w,cl):
    # cluster-robust weighted t by problem
    W=w.sum(); mu=(w[:,None]*D).sum(0)/W
    res=w[:,None]*(D-mu)
    S=np.zeros((len(uid),D.shape[1]),np.float32); np.add.at(S,cl,res)
    se=np.sqrt((S**2).sum(0))/W+1e-6
    return mu,mu/se
def null_count(D,w,cl,th=4,n=200,seed=3):
    r=np.random.default_rng(seed); out=[]
    for _ in range(n):
        sg=r.choice([-1.,1.],len(uid))[cl][:,None].astype(np.float32)
        _,t=wt(D*sg,w,cl); out.append((np.abs(t)>th).sum())
    return np.array(out)
res={}
for key in ['m11','m17','m23']:
    D=contrast(key).astype(np.float32)
    # remove length difference coordinate-wise using through-origin slope (descriptive adjustment only)
    b=(w[:,None]*dlen[:,None]*D).sum(0)/((w*dlen**2).sum()+1e-9)
    Da=D-dlen[:,None].astype(np.float32)*b[None]
    mu,t=wt(D,w,uidx); mua,ta=wt(Da,w,uidx)
    nc=null_count(Da,w,uidx)
    res[key]=(mu,t,mua,ta)
    print(key,'raw |t|>4: %d; length-adjusted |t|>4: %d; cluster sign-flip null median %d, 95pct %d, max %d; corr(t,slope) %.2f'%((np.abs(t)>4).sum(),(np.abs(ta)>4).sum(),np.median(nc),np.percentile(nc,95),nc.max(),np.corrcoef(t,b)[0,1]))
    top=np.argsort(-np.abs(ta))[:8]; print('   top',[(int(n),round(float(ta[n]),1)) for n in top])
```