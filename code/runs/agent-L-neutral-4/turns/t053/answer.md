A loop-onset direction exists from layer 16 onward. Its cross-validated effect size is d ≈ 1.2-1.3 at layers 16, 20 and 23, against 0.1 at layer 8 and 0.6 at layer 12, so it is built between layers 12 and 16. The onset decision is soft: median first-token logprob -0.21, and only 35% of onsets are near-certain. Inside a loop, copying is fully certain.

Before building an edit I have to rule out two confounds:
1. Position. Onsets come late in a text, and the direction might just encode "far into the answer".
2. Correct versus stuck.

So I'll recompute the separation within position bins, and compare where three groups project: novel newlines in finished texts, in-loop newlines, and novel newlines before onset in looping texts. I'll also check the cosine with the earlier style direction u. If d stays large within position bins and in-loop positions project high, the direction marks "about to repeat" and is a valid target. If d collapses within bins, it's a position artifact.

```python
L=20
H=np.load(f'/work/fwd_8/hidden_{L}.npy',mmap_mode='r')
v=dirs[L]/np.linalg.norm(dirs[L])
def proj(lst): return np.array([float(H[i,q].astype(np.float32)@v) for i,q in lst])
po,pn,pl=proj(onset),proj(novel),proj(inloop[::5])
qo=np.array([q-plen[i] for i,q in onset]); qn=np.array([q-plen[i] for i,q in novel])
bins=[0,100,200,300,450,800]; rows=[]
for b0,b1 in zip(bins[:-1],bins[1:]):
    a=po[(qo>=b0)&(qo<b1)]; b=pn[(qn>=b0)&(qn<b1)]
    if len(a)>5: rows.append(dict(bin=f'{b0}-{b1}',no=len(a),nn=len(b),d=round((a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2),2)))
fin_mask=np.array([meta[i]['src']=='g1fin' for i,_ in novel])
sd=pn.std()
print(pd.DataFrame(rows).to_string())
print('mean proj (in sd units, rel. novel-finished): onset',round((po.mean()-pn[fin_mask].mean())/sd,2),
      'inloop',round((pl.mean()-pn[fin_mask].mean())/sd,2),'novel-in-looping-texts',round((pn[~fin_mask].mean()-pn[fin_mask].mean())/sd,2))
uu=u/np.linalg.norm(u) if 'u' in dir() else None
print('cos with style u:', None if uu is None else round(float(v@np.asarray(uu,dtype=np.float32)),3))
print('cos across layers:',{l:round(float(dirs[l]@dirs[20]/np.linalg.norm(dirs[l])/np.linalg.norm(dirs[20])),2) for l in dirs})
```