This is a large jump: removing one direction from head 3 raises accuracy to 46.3% from 24.2%, far above anything else so far, and the tensor changed by only 6%. This was also the test I set up as decisive for the anchor-broadcast idea, so its success is meaningful: the harmful thing appears to be that specific fixed signal being read from the start of the model's turn and written through head 3, not the head as a whole.

I need to be careful with a number like this. It's still a `dev_a` figure, and although the edit wasn't tuned on these answers, I've now tried many variants on this set. Before believing it, I'll get the paired statistics, the split by failure type, and consistency across subsets. I'll also read a few answers to make sure they're genuine solutions, since a surprising gain could come from something odd. Then I'll confirm on fresh `dev_b` problems with a "both" pass, because that's the honest check for this one. If `dev_b` shows a similar gain, I'll go on to strengths and to whether the other heads carry the same signal.

```python
g18,c18,cu18=cmp_gen('/work/gen_18.jsonl','T2')
for nm_,c_ in [('H3k20',c14),('H3k40',c12)]:
    dd=c18.astype(float)-c_.astype(float); print('T2 vs',nm_,'paired %+.2f se %.2f'%(100*dd.mean(),100*dd.std(ddof=1)/np.sqrt(len(dd))))
d18=c18.astype(float)-c1.astype(float); print('fifths (pts):',[round(100*float(x.mean()),1) for x in np.array_split(d18,5)])
print('endings',dict(collections.Counter(ending(g18[i]['text']) for i in ids if not g18[i]['cut'])))
print('style top:',collections.Counter(first_words(g18[i]['text']) for i in ids).most_common(5))
print('p(answers contain repeated gold-free junk) check: mean tokens finished %.0f; unique first-40-char openings %d'%(np.mean([g18[i]['n_tokens'] for i in ids if not g18[i]['cut']]),len({g18[i]['text'][:40] for i in ids})))
wr_=[i for k,i in enumerate(ids) if c18[k] and not c1[k]]
for i in wr_[:3]:
    print('--',i,'gold',g18[i]['gold'],'orig pred',g1[i]['pred'],'T2 pred',g18[i]['pred'],'toks',g1[i]['n_tokens'],'->',g18[i]['n_tokens'])
    t_=g18[i]['text']; print(' T2 head:',sh(t_,170)); print(' T2 tail:',sh(t_.strip()[-110:],110))
os.makedirs('/work/best_5',exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/'+f_,'/work/best_5/'+f_)
open('/work/notes.md','a').write("""
pass 18 result T2 (head 3 du direction removed, a=1) dev_a acc .4629 cut .092; saved /work/best_5. HUGE vs H3k20 .3828 and baseline .2422.
Current edit T2 unchanged. pass 19 (planned): generate BOTH T2 on dev_b (fresh for this edit, 256 per model).
""")
```