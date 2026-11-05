It replicates. On p8, where earlier edits collapsed, L11 gives **+12.9 ± 3.2** (51 gained, 18 lost), with accuracy going from 23.4% to 36.3% and cut-offs from .238 to .168. That clears my predicted +8. Pooled over the 512 DEV problems, the gain is about **+14.3 ± 2.0**, more than five times the previous best (+2.5).

I won't touch HOLDOUT yet, because the protocol says to freeze the best candidate first, and I have good reason to tune on DEV before freezing. There are real open questions: whether all four L11 heads contribute (L11h2 hardly parked at all, at .05), how the effect responds to strength, and where the freed attention goes. The method also requires evidence at several strengths, which I don't have yet.

The first job is calibrating my cheap screens. The reference screen gave P1 only +1.31 and rejected L5 and the sliding layers, so I need to know if it can see a gain this large. I'll run Q11 unchanged through the packed fork and reference set. If both screens show a strong effect, they can tune strength and heads for one pass each. If they don't, they aren't valid proxies for accuracy, those rejections are unreliable, and I'll tune with generation only. I'll print the p8 breakdown and pooled figures first.

```python
arm_sum(29)
d=[]
for p in (28,29):
    e={r['id']:r for r in lg(f'/work/gen_{p}_edited.jsonl')}; o={r['id']:r for r in lg(f'/work/gen_{p}_original.jsonl')}
    d+= [int(e[k]['correct'])-int(o[k]['correct']) for k in e]
d=np.array(d,float); print('pooled n',len(d),'gain %.2f se %.2f'%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d))),'WR',int((d>0).sum()),'RW',int((d<0).sum()))
open('/work/notes.md','a').write(f"""
pass29 Q11 p8: +12.89+-3.15 (WR51 RW18), acc .234->.363, cut .238->.168. REPLICATED. Pooled p7+p8 n=512: {100*d.mean():.2f}+-{100*d.std(ddof=1)/np.sqrt(len(d)):.2f}.
Plan: tune on DEV before freezing (heads, strength, L2 combo, attention mechanism), reserve last 4 passes for HOLDOUT (900 ids, 4 both passes).
pass30 plan: teacher_forced EDITED (Q11 unchanged) /work/tf10_texts.json token_logprobs topk 10. Calibrate fork and ref screens against a known large gain.
""")
```