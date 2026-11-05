I'm glad I printed that before running anything: the automatic strength for h1 came out at **33.89, which changes that head by 166%**. That would have been a meaningless edit, and I'll discard it without spending a pass. The numbers explain why: h1's query has almost no component along the pure direction (0.13 logits of its 5.0 advantage). Its last-token advantage sits almost entirely in the part shared with BOS, so this direction can't remove it, and h1 only reads the last token at .02 anyway because BOS outcompetes it. So I'm leaving h1 out.

For h0, the formula's strength of 2.8 is also more than needed. h0 reads the last token at only .12, so strength 1.5 lowers the advantage by 2.45 logits, enough to take that share down to about .01. I'll use the same 1.5 on h0, h2 and h3, which keeps the edit uniform and avoids a large reversal on one head. I'll overwrite the bad candidate and log what happened, so that folder can't be picked up later by mistake.

```python
import shutil, os
shutil.rmtree('/work/cand_T_pure4',ignore_errors=True)
r=set_dirs({0:[(ph,1.5)],2:[(ph,1.5)],3:[(ph,1.5)]},'T_pure023')
Qs=load(nm); Qo=load_original(nm)
print(r, 'max rel head change', max(r[1]), 'h1 identical', bool((Qs[256:512]==Qo[256:512]).all()))
with open('/work/notes.md','a') as f:
    f.write("\nT_pure4 as logged above was a BUILD ERROR, never run: auto strength h1 33.89 (rel 1.66) because h1 q.ph tiny (A .13 of adv 5.0: its LAST advantage lies in BOS-shared part). Folder deleted. Pure direction cannot fix h1, h1 reads LAST only .02: h1 excluded.\nEDIT T_pure023 (current): ph beta 1.5 on heads 0,2,3, h1 original. saved /work/cand_T_pure023. h0 predicted LAST adv 5.18->2.73, share .12->~.01.\npass41 plan (replaces above): both p7. Baseline T_pure p7 +22.66.\n")
```