The pooled HOLDOUT result clears my pre-set bar. Over 1,226 never-tuned problems, C(−1.5) gives **+2.20 ± 0.95 points** (82 wrong→right, 55 right→wrong), 2.3 standard errors above zero. DEV pooled is +1.17 ± 1.38 (512 problems), smaller but the same sign. I've adopted C. It stays in /work/edited (14 tensors) with a copy in /work/best_final.

Before deciding how to use the remaining 17 passes, I want to check that the gain comes through the mechanism I targeted. If C works by breaking loops, the wrong→right flips should come mostly from problems where the original was cut off, and the right→wrong flips mostly from problems where it had finished correctly. This needs no pass: I'll cross-tabulate the HOLDOUT flips by the original's outcome. If most gains instead come from problems the original finished wrongly, the mechanism is something other than loop breaking and I'd need to look more closely.

```python
tab=defaultdict(lambda:[0,0,0])
for k in [29,30,31,32,33]:
    o={json.loads(l)['id']:json.loads(l) for l in open(f'/work/gen_{k}_original.jsonl')}
    e={json.loads(l)['id']:json.loads(l) for l in open(f'/work/gen_{k}_edited.jsonl')}
    for i in o:
        st='orig_cut' if o[i]['cut'] else ('orig_right' if o[i]['correct'] else 'orig_wrong_fin')
        dd=int(e[i]['correct'])-int(o[i]['correct'])
        t=tab[st]; t[0]+=1; t[1]+= dd>0; t[2]+= dd<0
        if o[i]['cut']:
            t2=tab['orig_cut->edited_'+('cut' if e[i]['cut'] else 'fin')]; t2[0]+=1; t2[1]+=dd>0
{k:tuple(v) for k,v in tab.items()}
```