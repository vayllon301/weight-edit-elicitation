The positions are confirmed. Texts alternate A, B. In each, the last token is the newline, the k result digits sit just before it, then a space, then " =". I'll take the space token, which predicts the first result digit, along with the "=" token and the last digit position. For each layer I'll compute the following A−B statistics: the relative size of the mean difference against the residual norm, the mean cosine of per-pair differences with the mean difference (consistency), and the per-layer increment of the mean difference. I'll also record per-expression whether A got the result wrong, to see whether the difference direction separates wrong from right A cases.

```python
A=np.arange(0,374,2); B=A+1
def positions(i):
    n=L[i]; s=texts[i].rstrip('\n'); k=len(s.split('=')[-1].strip())
    return n,k
P={}
for i in range(374):
    n,k=positions(i); P[i]=(n-2-k, n-3-k, n-2)  # space, '=', last digit
# correctness of result in each text
def res_lp(i):
    n,k=positions(i); return float(sum(tlp[i,j-1] for j in range(n-1-k,n-1)))
lpA=np.array([res_lp(i) for i in A]); lpB=np.array([res_lp(i) for i in B])
print('mean result lp A %.3f B %.3f; frac lp<-0.7: A %.2f B %.2f'%(lpA.mean(),lpB.mean(),(lpA<-0.7).mean(),(lpB<-0.7).mean()))
rows=[]
prev=None
for l in range(27):
    H=np.load(d+f'hidden_{l}.npy',mmap_mode='r')
    ha=np.stack([H[i,P[i][0]].astype(np.float32) for i in A]); hb=np.stack([H[i,P[i][0]].astype(np.float32) for i in B])
    D=ha-hb; m=D.mean(0)
    rn=np.linalg.norm(hb,axis=1).mean()
    cons=np.mean((D@m)/(np.linalg.norm(D,axis=1)*np.linalg.norm(m)+1e-9))
    inc=np.nan if prev is None else np.linalg.norm(m-prev)/rn
    # does projection on m separate wrong vs right A?
    proj=(ha@m)/np.linalg.norm(m)
    wrong=lpA<-0.7
    sep=(proj[wrong].mean()-proj[~wrong].mean())/(proj.std()+1e-9) if wrong.sum()>2 else np.nan
    rows.append((l, np.linalg.norm(m)/rn, np.linalg.norm(D,axis=1).mean()/rn, cons, inc, sep))
    prev=m
    np.save(f'/work/mAB_space_{l}.npy',m)
for r in rows: print('L%2d |mean|/rn %.3f |D|/rn %.3f cons %.2f inc %.3f sepWrong %.2f'%r)
```