The last HOLDOUT quarter is flat: accuracy went from 9.6% to 10.0%, a paired +0.4 ± 2.2 points (15 wrong→right, 14 right→wrong), with cut-offs up from 29 to 50.

Summing the four pass summaries over all 1,000 HOLDOUT problems, the original gets about 113 right and E18 about 102, roughly −1.1 points. I'll compute the exact pooled paired difference and its standard error from the four answer files. Then I'll drop E18 for good and confirm /work/edited is empty, since by the pre-registered rule the original is the submission.

```python
d=[]; co=ce=0; cut_o=cut_e=0
for k in (45,46,47,48):
    o={r['id']:r for r in map(json.loads,open(f'/work/gen_{k}_original.jsonl'))}
    e={r['id']:r for r in map(json.loads,open(f'/work/gen_{k}_edited.jsonl'))}
    for i in o:
        d.append(int(e[i]['correct'])-int(o[i]['correct'])); co+=o[i]['correct']; ce+=e[i]['correct']; cut_o+=o[i]['cut']; cut_e+=e[i]['cut']
d=np.array(d); m=d.mean()*100; se=d.std(ddof=1)/np.sqrt(len(d))*100
drop_edit('model.layers.9.self_attn.o_proj.weight')
with open('/work/notes.md','a') as f:
    f.write(f"pass 48 done: E18 HOLDOUT q3: 9.6% -> 10.0%, +0.4+-2.2. POOLED HOLDOUT {len(d)}: orig {co} -> E18 {ce} correct, paired {m:+.2f} +- {se:.2f} pts; cuts {cut_o}->{cut_e}. E18 REJECTED, dropped. FINAL SUBMISSION = ORIGINAL (no edits).\n")
print(len(d), co, ce, round(m,2), round(se,2), cut_o, cut_e, (d==1).sum(), (d==-1).sum(), edited())
```