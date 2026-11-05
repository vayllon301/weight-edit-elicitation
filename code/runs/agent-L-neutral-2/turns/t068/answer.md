The traceback is my indexing bug: I scored only half B but then indexed it with whole-set positions. It's easy to fix, and the logit comparison above it came through fine.

There's a real partial signal in Let style. Answers with an unused number are about half as confident at the decision to end the reasoning: the median margin for the final line break is 1.38 against 3.62 for covered answers, with AUC around 0.64-0.67. Here-style shows almost nothing, at 0.53-0.57, but my measure is flawed there. Its competitor at the stop line is often the *other* ending (`####` or `**`), so that margin isn't really stop versus continue. I need to count both endings as stopping, and I'll rely on the line-break decision for Here-style.

I'll rerun the hidden-state check with the fix, and print how many covered and uncovered answers fall under a few margin levels. That shows how selective a small push against stopping could be. If uncovered answers are much more common at low margins, a nudge affects them first. If the two groups overlap heavily, a nudge would also disturb the correct answers, which have the same margins as covered wrong ones.

```python
rng=np.random.default_rng(1)
labs=K.miss.values; sty_=(K.sty=='Here').values
for l in [16,18,22,24,26]:
    Hh=np.load(d+f'hidden_{l}.npy',mmap_mode='r')
    Xs=np.stack([np.asarray(Hh[i,K.p[i]],dtype=np.float32) for i in range(len(K))]).astype(np.float64)
    Xp=np.stack([np.asarray(Hh[i,K.p[i]-1],dtype=np.float32) for i in range(len(K))]).astype(np.float64)
    res=[]
    for nm,XX in [('stop-pos',Xs),('nl-pos',Xp)]:
        a=collections.defaultdict(list)
        for rep in range(40):
            perm=rng.permutation(len(K)); A=perm[:len(K)//2]; B=perm[len(K)//2:]
            dd=np.zeros(1152)
            for s_ in (0,1):
                ia=A[sty_[A]==s_]; dd+=XX[ia[labs[ia]]].mean(0)-XX[ia[~labs[ia]]].mean(0)
            sc=XX@dd
            for s_ in (0,1):
                ib=B[sty_[B]==s_]
                if labs[ib].sum()>2: a[s_].append(roc_auc_score(labs[ib],sc[ib]))
        res.append(f"{nm}: Let {np.mean(a[0]):.3f} Here {np.mean(a[1]):.3f}")
    print('hidden',l,' | '.join(res))
for s_ in ['Let','Here']:
    q=K[K.sty==s_]
    for lab,sel in [('correct',q.correct),('wrong covered',(~q.correct)&(~q.miss)),('wrong unused',(~q.correct)&q.miss)]:
        v=q[sel].nl_margin.values
        print(s_,f'{lab:14s} n {len(v):3d} nl margin quantiles 10/25/50/75: ',np.percentile(v,[10,25,50,75]).round(2),'share<1 %.2f <2 %.2f <3 %.2f'%((v<1).mean(),(v<2).mean(),(v<3).mean()))
```