With batching reordered, C(−1.5) on holdout_1 gives +4.3 ± 2.2 (22 wrong→right, 11 right→wrong), against +5.9 ± 2.2 (24/9) in pass 29. That is the same sign at a similar size. Accuracies are 18.8% → 23.0%, against 18.4% → 24.2%.

Next I'll measure the per-problem reproducibility:
- for each model, the share of problems with the same grade in both passes, and the share with identical text;
- how many of pass 29's flips recur in pass 48.

This quantifies the batching noise the evaluation will also see.

```python
def ld(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
o29,e29,o48,e48=[ld(f'/work/gen_{k}_{m}.jsonl') for k,m in [(29,'original'),(29,'edited'),(48,'original'),(48,'edited')]]
ids=list(o29)
same=lambda A,B,f: np.mean([A[i][f]==B[i][f] for i in ids])
f29=[i for i in ids if o29[i]['correct']!=e29[i]['correct']]; f48=[i for i in ids if o48[i]['correct']!=e48[i]['correct']]
w29={i for i in f29 if e29[i]['correct']}; r29=set(f29)-w29; w48={i for i in f48 if e48[i]['correct']}; r48=set(f48)-w48
res=dict(orig_same_grade=round(same(o29,o48,'correct'),3),orig_same_text=round(same(o29,o48,'text'),3),
         edit_same_grade=round(same(e29,e48,'correct'),3),edit_same_text=round(same(e29,e48,'text'),3),
         w2r_29=len(w29),w2r_48=len(w48),w2r_both=len(w29&w48),r2w_29=len(r29),r2w_48=len(r48),r2w_both=len(r29&r48))
with open('/work/notes.md','a') as f: f.write(f"Pass48 result C(-1.5) both holdout_1 REVERSED (reproducibility only, not pooled): +4.30+-2.23 (22/11) vs pass29 +5.86 (24/9). Repro: {res}\n")
res
```