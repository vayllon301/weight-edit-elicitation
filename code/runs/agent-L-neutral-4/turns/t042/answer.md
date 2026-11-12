Block 32 measured copied queries only; its novel reference (a fixed offset of q-30) was never reported. Now I'll compute a fair comparison with one definition for every query: the 1-gram induction target. That is the attention from query q to position p+1, where p is the most recent earlier position holding the same token as q. I'll average it separately over copied queries (whose 8-gram occurred before) and over novel queries. I'll also report each head's ratio and its share of total attention to recent positions.

Some heads should show high induction mass on copied queries but low mass on novel ones. Those are loop-specific candidates. Heads that score similarly on both are generic induction heads that normal reasoning needs too (for example, copying numbers from the problem).

```python
res=[]
for layer in [5,11,17,23]:
    A=np.load(f'/work/fwd_7/attn_{layer}.npy', mmap_mode='r')
    for h in range(4):
        ic=[];inov=[];ic8=[]
        for i in range(2):
            a=ia[i,:L7[i]]; Ah=np.asarray(A[i,h,:L7[i],:L7[i]],dtype=np.float32)
            last8={}; last1={}
            for q in range(7,L7[i]-1):
                ng=tuple(a[q-7:q+1]); t=a[q]
                copied = ng in last8
                if q>100 and t in last1:
                    v=Ah[q,last1[t]+1]
                    (ic if copied else inov).append(v)
                if q>100 and copied: ic8.append(Ah[q,last8[ng]+1])
                last8[ng]=q; last1[t]=q
        res.append(dict(layer=layer,h=h,ind8_copy=round(float(np.mean(ic8)),3),ind1_copy=round(float(np.mean(ic)),3),
                        ind1_novel=round(float(np.mean(inov)),3),ratio=round(float(np.mean(ic)/max(np.mean(inov),1e-4)),1),nc=len(ic),nn=len(inov)))
pd.DataFrame(res).to_string()
```