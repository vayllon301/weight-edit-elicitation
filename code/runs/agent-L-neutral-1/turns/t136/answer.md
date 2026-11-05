The edit is saved, but the notes entry failed on a formatting character, and the rule is that notes come before any pass, so I'll fix that first. `loop_metrics` failed for a similar reason: it reads a global called `rows`, which I overwrote with a list in an earlier run. I'll restore that and list every global these metric functions depend on, so nothing else can silently use the wrong data.

The summary check shows the edit does what I designed, but it's less clean than I'd hoped. Mean queries are preserved exactly, and the loop shift falls in most heads (L5h2 .445→.309, L11h1 .310→.213). However, L11h2 rises from .26 to .31, because `x33` was partly cancelling other coordinates there, and L11h3 barely moves. Prompt-token queries also change by .15–.39 of their rms, which is as large as the state shift I'm removing, so damage to prompt reading is a real risk. That makes the digit results on reference solutions as important as the loop results in this test.

After this run, I'll have a verified baseline and a complete notes entry for the pass.

```python
rows=lr
print('loop_metrics names',loop_metrics.__code__.co_names)
print('digit_metrics names',digit_metrics.__code__.co_names)
try:
    out=loop_metrics(e17,b7,0); print('loop_metrics(pass17,0):',str(out)[:600])
except Exception as ex_: print('loop_metrics err',repr(ex_))
note=("\nReuse/ex analysis (no pass): numbers used as ref (ex=mi=0) acc .445; ex>0 acc .076, flat in steps; adjacent same-op replay rarer in own than ref => ex is comprehension, not eq replay. CLOSED.\n"
"X33 STATE: raw x33 prompt<pre<c1<later at L5,L11,L17. q col33 top norm at L4,L5,L11 (input_ln gain ~11). Largest single contributor to pre->later query shift L5 15-48pct, L11 19-78pct; L17/L23 are 350,367,775. q33.k33 cos L11 all heads .19-.37; L5 only h2 .36 (replay head, largest shift .45). Feedback not specific to replay heads overall. Hypothesis: x33 gains prompt-vs-self preference.\n"
"pass21: TF. EDIT cand_x33_b0: L5 and L11 q_proj all heads: col33*=0, col1038 += col33_orig*r (r=x33/x1038 after-norm pre-loop means, L5 {:.3f}, L11 {:.3f}; rel change .045,.081). Summary check: mean query kept; loop shift mostly down (L11h2 up .26->.31); prompt queries change .15-.39 rms = main risk. texts=/work/tf17_texts.json (loops+refs+own), return as pass17, baselines pass7, pass2. Prediction: loops leave cycle more, digits little changed. Digit loss on refs would refute usefulness.\n").format(EDIT[5][2],EDIT[11][2])
with open('/work/notes.md','a') as fh: fh.write(note)
print(open('/work/notes.md').read()[-420:])
print('saved cand folder exists',os.path.isdir('/work/cand_x33_b0'))
os.makedirs('/work/cand_x33_b0',exist_ok=True)
for fn_ in os.listdir('/work/edited'): shutil_src='/work/edited/'+fn_; open('/work/cand_x33_b0/'+fn_,'wb').write(open(shutil_src,'rb').read())
print('cand files',os.listdir('/work/cand_x33_b0'),'edited',edited())
m17=json.load(open('/work/fwd_17/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else '...') for k,v in m17.items()}); print('fwd_17 files',sorted(os.listdir('/work/fwd_17')))
```