The positive control passed. The comparator reproduces the logged DEV table for the edit (copy first +1.455, reuse first −0.951, result later +1.315), so the pipeline is intact. The format is also settled: 512 reference texts are the annotation-stripped solution, a newline, then `#### <answer>`, with nothing after. So the HOLDOUT measurement can be read directly against DEV.

Now I'll build the same texts for all 1,024 HOLDOUT problems and run two more checks:
- **Format:** a DEV text rebuilt with this builder must match its stored text exactly.
- **Net figure:** my net calculation must return the logged +0.51 ± 0.13 on pass 39.

The success criteria go in the notes before any HOLDOUT teacher-forced data exists. If both checks pass, pass 49 runs the original model on these texts and pass 50 runs the frozen edit on the same file.

```python
ho=json.load(open('/work/holdout_ids.json'))
byid={r['id']:r for r in rec}
build=lambda r: PROMPT(r['problem'])+clean(r['solution']).strip()+'\n#### '+r['answer']
T_h=[build(byid[i]) for i in ho]
devset=set(t17)
dv=json.load(open('/work/dev_ids.json'))
n_dev=sum(build(byid[i]) in devset for i in dv)
print('dev rebuilt that match stored texts',n_dev,'of 512 refs; hold',len(T_h),'in dev texts',sum(x in devset for x in T_h))
L=[len(tok.encode(x).ids) if hasattr(tok.encode(x),'ids') else len(tok.encode(x)) for x in T_h]
print('tokens total',sum(L),'max',max(L),'median',int(np.median(L)))
def net(rows):
    g=rows.assign(e=rows.top_e.astype(float),b=rows.top_b.astype(float),dd=rows.top_e.astype(float)-rows.top_b.astype(float)).groupby('t')
    s=g['dd'].sum(); n=g['dd'].count(); m=s.sum()/n.sum()
    return 100*m,100*np.sqrt(((s-m*n)**2).sum())/n.sum(),int(n.sum())
T_cur=t17
rows39,_=digit_table(39,sel,base,kbase=2)
print('control pass39 net %.3f se %.3f n %d (logged +.51+-.13)'%net(rows39))
T_cur=_T
json.dump(T_h,open('/work/tf49_texts.json','w'))
with open('/work/notes.md','a') as f:
    f.write(f"""
Comparator control OK: digit_table(39,sel,base,kbase=2) reproduces logged table. Ref format = PROMPT + solution minus <<..>> stripped + newline + '#### ' + answer.
SPARE PASSES (cannot change submission): HOLDOUT mechanism test. texts=/work/tf49_texts.json (all 1024 HOLDOUT refs, order holdout_ids), token_logprobs, topk 10. Compare with digit_table(50,range(1024),range(1024),kbase=49), T_cur=T_h.
PRE-SET: mechanism CONFIRMED if net digit top1 >= +.25 and z>2, with copy first >0 and reuse first <0. REFUTED if net near 0 (z<1). In between = partial. Either way submission stays cand_num_p1.
pass49: TF ORIGINAL on /work/tf49_texts.json. Edit on disk unchanged (cand_num_p1), not used by this pass.
pass50: TF EDITED, same file, cand_num_p1 frozen.
""")
print('saved')
```