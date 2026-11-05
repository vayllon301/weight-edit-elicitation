The pass finished with the edit applied, so I'll score it exactly as registered. I'll report the primary set first, with exact McNemar on the paired right/wrong outcomes and the paired `lp` change, then the k=2 control and frame Q. The breakdowns by operation and term count are descriptive only. If the primary set is flat, a gain in some subgroup won't rescue it, because I'd be picking that subgroup after seeing the data.

```python
I46=np.load('/work/fwd_46/input_ids.npy',mmap_mode='r'); TK46=np.load('/work/fwd_46/topk_ids.npy',mmap_mode='r'); TL46=np.load('/work/fwd_46/topk_logprobs.npy',mmap_mode='r'); LP46=np.load('/work/fwd_46/token_logprobs.npy',mmap_mode='r')
print('same input ids', bool(np.array_equal(np.asarray(I46),np.asarray(I))))
e46=score(TK46,LP46,I46,ixc)
e46.to_pickle('/work/chain_edit46.pkl')
def rep(mask,name):
    o=base[mask]; e=e46[mask]
    n=len(o); bo=int((e.ok.values&~o.ok.values).sum()); ob=int((o.ok.values&~e.ok.values).sum())
    p=st.binomtest(bo,bo+ob,.5).pvalue if bo+ob>0 else 1.0
    fb=int((e['first'].values&~o['first'].values).sum()); fo=int((o['first'].values&~e['first'].values).sum())
    pf=st.binomtest(fb,fb+fo,.5).pvalue if fb+fo>0 else 1.0
    dl=e.lp.values-o.lp.values
    print(f"{name:26s} n {n:3d} ok {o.ok.mean():.3f}->{e.ok.mean():.3f} (w->r {bo}, r->w {ob}, p {p:.2g}) first {o['first'].mean():.3f}->{e['first'].mean():.3f} ({fb}/{fo}, p {pf:.2g}) lp {o.lp.mean():.2f}->{e.lp.mean():.2f} d {dl.mean():+.2f}+/-{dl.std(ddof=1)/np.sqrt(n):.2f}")
aS=base.op.isin(['add','sub']); HL=base.frame.isin(['H','L'])
rep(aS&HL&(base.k>=3),'PRIMARY H+L k>=3')
rep(aS&HL&(base.k==2),'CONTROL H+L k=2')
rep(aS&(base.frame=='Q'),'SECONDARY Q all k')
print('--- descriptive only')
for fr in 'HLQ':
    for op,ks in (('add',[2,3,4,5]),('sub',[2,3,4])):
        for k in ks:
            rep((base.frame==fr)&(base.op==op)&(base.k==k),f'{fr} {op} k{k}')
rep(base.op.isin(['mul1','mul2']),'excluded mixed')
```