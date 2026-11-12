On holdout_3 reversed, C(−1.5) gives +0.78 ± 2.07 (15/13), against −0.78 (15/17) in pass 31. Both are near zero, differing by 4 right→wrong cases. Cut-offs fall from 24% to 12.5%, again as intended. I'll compute the same per-problem reproducibility numbers against pass 31 and record them. Then I'll set up the last pass as the third reproducibility check, on holdout_2 reversed (pass 30 measured +4.3, 17/6). Like the others, it will not be pooled into the gain.

```python
o31,e31,o49,e49=[ld(f'/work/gen_{k}_{m}.jsonl') for k,m in [(31,'original'),(31,'edited'),(49,'original'),(49,'edited')]]
ids=list(o31)
f31=[i for i in ids if o31[i]['correct']!=e31[i]['correct']]; f49=[i for i in ids if o49[i]['correct']!=e49[i]['correct']]
w31={i for i in f31 if e31[i]['correct']}; r31=set(f31)-w31; w49={i for i in f49 if e49[i]['correct']}; r49=set(f49)-w49
res3=dict(orig_same_grade=round(np.mean([o31[i]['correct']==o49[i]['correct'] for i in ids]),3),
          edit_same_grade=round(np.mean([e31[i]['correct']==e49[i]['correct'] for i in ids]),3),
          w2r=(len(w31),len(w49),len(w31&w49)),r2w=(len(r31),len(r49),len(r31&r49)))
ids_h2=json.load(open('/work/holdout_2.json')); json.dump(ids_h2[::-1],open('/work/holdout_2_rev.json','w'))
with open('/work/notes.md','a') as f:
    f.write(f"Pass49 result C(-1.5) both holdout_3 REVERSED (repro only): +0.78+-2.07 (15/13) vs pass31 -0.78 (15/17). Repro (pass31,pass49,both): {res3}\n"
            "Current edit: C(-1.5) (=best_final, submission). Pass 50 (planned, last): generate BOTH on /work/holdout_2_rev.json -> repro of pass30 (+4.30, 17/6); not pooled.\n")
(res3, len(ids_h2))
```