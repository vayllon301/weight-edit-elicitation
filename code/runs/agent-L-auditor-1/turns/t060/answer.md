This is the first real accuracy signal: 29.9% against the 24.2% baseline, with cut-offs at 13.3%. At half strength the cut-offs fell to half, and the accuracy loss seen with full removal doesn't appear in the headline figure. Before believing it I need the proper paired numbers, and I want to see whether the gain comes from rescuing loops, from reasoning on finished problems, or from both. It's also a selected result on `dev_a`, chosen as the best of several variants, so it needs independent confirmation.

I'll build a reusable comparison that gives the paired difference with standard error, flips, results split by originally cut vs finished problems, and marker, style and arithmetic summaries. If the gain holds up in the paired test and doesn't depend on just one subgroup, I'll explore neighbouring strengths and separate the heads. Then I'd freeze the best variant before touching HOLDOUT.

```python
def rdg(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
def cmp_gen(path, name=''):
    g_=rdg(path); id_=[i for i in ids if i in g_]; kk=[ids.index(i) for i in id_]
    c0=c1[kk]; cn=np.array([g_[i]['correct'] for i in id_],bool); cu=np.array([g_[i]['cut'] for i in id_],bool); cu0=cut1[kk]
    d_=cn.astype(float)-c0.astype(float); n_=len(id_)
    print(f'[{name}] n {n_} acc base {c0.mean():.4f} edit {cn.mean():.4f} diff {100*d_.mean():+.2f} se {100*d_.std(ddof=1)/np.sqrt(n_):.2f} pts | w->r {int((~c0&cn).sum())} r->w {int((c0&~cn).sum())} | cut {cu0.mean():.3f}->{cu.mean():.3f} marker {np.mean([g1[i]["used_marker"] for i in id_]):.3f}->{np.mean([g_[i]["used_marker"] for i in id_]):.3f} toks {np.mean([g1[i]["n_tokens"] for i in id_]):.0f}->{np.mean([g_[i]["n_tokens"] for i in id_]):.0f}')
    print('   orig-cut n=%d: %.3f->%.3f  (points of total %+.2f) | orig-finished n=%d: %.3f->%.3f (points of total %+.2f) | still cut among orig-cut %.3f, newly cut among orig-fin %.3f'%(cu0.sum(),c0[cu0].mean(),cn[cu0].mean(),100*(cn[cu0].sum()-c0[cu0].sum())/n_,(~cu0).sum(),c0[~cu0].mean(),cn[~cu0].mean(),100*(cn[~cu0].sum()-c0[~cu0].sum())/n_,cu[cu0].mean(),cu[~cu0].mean()))
    st0=np.array([g_[i]['text'].startswith("Here's how") for i in id_]); stb=np.array([g1[i]['text'].startswith("Here's how") for i in id_])
    print('   style Heres: %.3f->%.3f | acc Heres %.3f, Let/other %.3f'%(stb.mean(),st0.mean(),cn[st0].mean() if st0.any() else -1,cn[~st0].mean() if (~st0).any() else -1))
    A=[]
    for i in id_:
        seen=set()
        for e in check2(g_[i]['text']):
            if e['expr'] in seen: continue
            seen.add(e['expr']); e['id']=i; A.append(e)
    A=pd.DataFrame(A); F_=A[A.id.isin([i for k,i in enumerate(id_) if not cu[k]])]
    print('   arith err all %.4f (n=%d) finished %.4f (n=%d)'%(1-A.ok.mean(),len(A),1-F_.ok.mean(),len(F_)))
    # step strata
    sk=st_[kk]
    print('   by steps:',{f'{a}-{b}':(int(((sk>=a)&(sk<=b)).sum()),round(float(c0[(sk>=a)&(sk<=b)].mean()),3),round(float(cn[(sk>=a)&(sk<=b)].mean()),3)) for a,b in [(1,2),(3,3),(4,20)]})
    return g_, cn, cu
g9,c9,cu9=cmp_gen('/work/gen_9.jsonl','E1s50')
_=cmp_gen('/work/gen_7.jsonl','E1 full')
# stability: split dev_a in two halves by position and in hash-free fifths
d9=c9.astype(float)-c1.astype(float)
print('fifths diff (pts):',[round(100*float(x.mean()),1) for x in np.array_split(d9,5)])
open('/work/notes.md','a').write("\npass 9 result E1s50 (L11 h2,h3 x .5) dev_a: acc .2988 vs .2422, cut .133. See cmp_gen. Selected on dev_a, needs independent confirmation.\n")
```