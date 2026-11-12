On dev_f, E6 gives +3.9 ± 2.6, against +3.1 for E5. All three candidates were measured on the same two DEV sets:

| Candidate | dev_e | dev_f | Pooled (512 paired answers) |
|---|---|---|---|
| E4 | +3.1 | +1.6 | +2.3 |
| E5 | +5.5 | +3.1 | +4.3 |
| E6 | +8.6 | +3.9 | +6.3 |

E6 beats E5 on both sets, so I'm freezing E6, copying it to /work/best_4, and measuring it on the same four HOLDOUT sets, where E5 pooled +3.1 ± 1.3.

Measuring E6 now secures a confirmed candidate before any further exploration of layers 22–23. A pooled HOLDOUT gain above E5's would confirm the layer-18 extension. One no higher than E5's would mean the DEV improvement was selection noise, and I'd keep E5.

```python
os.makedirs('/work/best_4',exist_ok=True)
for f in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f,'/work/best_4/'+f)
open('/work/notes.md','a').write("\nPass 27 result: E6 dev_f +3.9+-2.6 (27 W->R,17 R->W), cuts 19->32. Pooled DEV dev_e+dev_f: E6 +6.25 vs E5 +4.3 vs E4 +2.3 (same problems).\nE6 FROZEN, copied to /work/best_4. Passes 28-31 plan: E6 'both' on /work/hold_0..3.json.\n")
os.listdir('/work/best_4')
```