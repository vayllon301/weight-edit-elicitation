Pass 6's attention covers 80 digit positions across two answers, enough for a first read on where each head looks. For each head at L11 and L17, at every position where the next token is a digit, I'll measure the attention mass on four groups: source positions holding exactly that next digit, other digits, the current token, and BOS. The key number is selectivity, the share of digit-directed attention that lands on the correct digit, compared with chance for that context.

An L11h3 with high selectivity for the correct next digit would be directly suppressing the right answer, which would make it a real candidate brake. If its attention falls mainly on the current token or on wrong digits, it's closer to a do-not-repeat mechanism. I'll print L11h2 alongside it, because the question is whether those two opposed heads read the same places. With two texts I can only treat this as a guide for designing a test.

```python
def attn_stats(l):
    A=np.load(f'/work/fwd_6/attn_{l}.npy',mmap_mode='r')
    out={h:[] for h in range(4)}
    for r in range(2):
        v=ids6[r]; n=int((v>=0).sum()); st=int(np.where(v==sot)[0][-1])
        isd=np.isin(v[:n],dset)
        for t in range(st+2,n-1):
            nx=int(v[t+1])
            if nx not in set(dset.tolist()): continue
            src=np.arange(1,t+1)
            same=(v[src]==nx); oth=isd[src]&~same
            base=same.sum()/max(1,(same.sum()+oth.sum()))
            for h in range(4):
                w=np.nan_to_num(np.asarray(A[r,h,t,:t+1],dtype=np.float64))
                out[h].append((w[0],w[t],w[src][same].sum(),w[src][oth].sum(),base,float(isd[t]),w[src][isd[src]].sum()-w[t]*isd[t]))
    return {h:np.array(x) for h,x in out.items()}
for l in (11,17,5):
    S=attn_stats(l)
    print('L%d  n=%d | h: BOS self correct-digit other-digit | selectivity(correct/(correct+other)) vs chance | mean eff'%(l,len(S[0])))
    for h in range(4):
        X=S[h]; c=X[:,2].sum(); o=X[:,3].sum()
        print(h,'%.3f %.3f %.3f %.3f | %.2f vs %.2f'%(X[:,0].mean(),X[:,1].mean(),X[:,2].mean(),X[:,3].mean(),c/(c+o+1e-12),X[:,4].mean()))
s11=attn_stats(11)
print('corr of correct-digit mass, h2 vs h3, per position %.2f; other-digit mass %.2f'%(np.corrcoef(s11[2][:,2],s11[3][:,2])[0,1],np.corrcoef(s11[2][:,3],s11[3][:,3])[0,1]))
A=np.load('/work/fwd_6/attn_11.npy',mmap_mode='r')
cs=[];
for r in range(2):
    n=int((ids6[r]>=0).sum()); st=int(np.where(ids6[r]==sot)[0][-1])
    for t in range(st+2,n-1):
        a=np.nan_to_num(np.asarray(A[r,2,t,1:t+1],dtype=np.float64)); b=np.nan_to_num(np.asarray(A[r,3,t,1:t+1],dtype=np.float64))
        if a.std()>0 and b.std()>0: cs.append(np.corrcoef(a,b)[0,1])
print('h2 vs h3 attention pattern corr per target position: median %.2f, q10 %.2f, q90 %.2f'%(np.median(cs),np.quantile(cs,.1),np.quantile(cs,.9)))
```