Keeping 20% of head 3 reaches 38.3%, about 14 points above baseline, and the best result yet. The head 3 curve now reads 24.2% at 100%, 33.8% at 40%, 38.3% at 20%, and 32.4% at 0%. I'm treating that peak carefully: I've compared several variants on the same 512 problems, each with a standard error of about 2 points. The jump between 40% and 20% might include some luck, and the highest point in a series tends to be overestimated. So the true value at 20% is probably below 38.3%, though still clearly above baseline.

I'll check the paired details. Then I'll separate tuning from verification by bringing in a fresh DEV subset, `dev_b`. HOLDOUT stays reserved for the final frozen candidate. On `dev_b` I'll use "both" passes with 256 problems per model so that comparisons are paired inside a single pass, as the protocol suggests. I'd first like one more `dev_a` point at 10% to outline the peak, but comparing 20% against 40% on fresh data is more useful. So the next step is a both-pass of 20% on a fresh 256 problems, and I'll look at its size relative to the `dev_a` figure. If the fresh gain is near +14, the peak is real. Something near +8 means `dev_a` overstated it, which wouldn't be surprising.

```python
g14,c14,cu14=cmp_gen('/work/gen_14.jsonl','H3k20')
for nm_,c_ in [('k40',c12),('k0',c13)]:
    dd=c14.astype(float)-c_.astype(float); print('H3k20 vs',nm_,'paired %+.2f se %.2f'%(100*dd.mean(),100*dd.std(ddof=1)/np.sqrt(len(dd))))
d14=c14.astype(float)-c1.astype(float); print('fifths (pts):',[round(100*float(x.mean()),1) for x in np.array_split(d14,5)])
print('endings',dict(collections.Counter(ending(g14[i]['text']) for i in ids if not g14[i]['cut'])))
os.makedirs('/work/best_4',exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+f_,'/work/best_4/'+f_)
dev=json.load(open('/work/dev_ids.json')); hold=json.load(open('/work/holdout_ids.json'))
rest=[i for i in dev if i not in set(ids)]
print('DEV',len(dev),'rest',len(rest),'overlap with holdout',len(set(dev)&set(hold)))
dev_b=rest[:256]; json.dump(dev_b,open('/work/dev_b.json','w'))
print('dev_b',len(dev_b),'overlap dev_a',len(set(dev_b)&set(ids)),'overlap holdout',len(set(dev_b)&set(hold)))
open('/work/notes.md','a').write("""
pass 14 result H3k20 (L11 head 3 x .2) dev_a acc .3828 cut .066; saved /work/best_4. Head-3 dose: k100 24.2, k40 33.8, k20 38.3, k0 32.4. Peak likely inflated by selection.
Current edit H3k20: model.layers.11.self_attn.o_proj.weight columns of head 3 x 0.2, heads 0,1,2 untouched.
dev_b = next 256 of DEV after dev_a, /work/dev_b.json (tuning, not holdout).
pass 15 (planned): generate BOTH H3k20 on dev_b (256 per model).
""")
```