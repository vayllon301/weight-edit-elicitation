On dev_e, E5 gives +5.5 ± 2.6 (29 answers went wrong to right, 15 right to wrong), against E4's +3.1 on the same problems. The two candidates were compared on the same two sets:

| Candidate | dev_e | dev_f | Pooled (512 paired answers) |
|---|---|---|---|
| E4 (n3729 zeroed) | +3.1 | +1.6 | +2.3 ± ~1.7 |
| E5 (n3729 reversed) | +5.5 | +3.1 | +4.3 ± ~1.7 |

E5 is ahead on both sets, so I'm freezing E5 as my candidate. It reverses the six layer-16 and two layer-17 doubt-neuron columns and the layer-16 n3729 column.

With 31 passes left, I'll confirm it now on HOLDOUT, split into four disjoint sets of 256 (1,024 of the 1,050 problems), pooling all four. That leaves about 27 passes for further DEV work. Any later candidate would have to beat E5 on DEV before getting its own HOLDOUT test.

Confirmation means a pooled HOLDOUT gain clearly positive, about +3 or more with SE ~1.2. A gain near zero would mean the DEV gains were partly selection noise.

```python
H=json.load(open('/work/holdout_ids.json'))
for k in range(4): json.dump(H[256*k:256*(k+1)], open(f'/work/hold_{k}.json','w'))
open('/work/notes.md','a').write("""
Pass 19 result: E5 on dev_e: +5.5+-2.6 (29 W->R,15 R->W), cuts 17->28.
 Pooled DEV (dev_e+dev_f, 512): E5 +4.3 vs E4 +2.3 (same problems). E5 FROZEN as candidate (best_2).
CURRENT EDIT E5. Pass 20-23 plan: 'both' E5 on HOLDOUT /work/hold_0..3.json (256 each), pooled.
""")
print(len(H), [len(json.load(open(f'/work/hold_{k}.json'))) for k in range(4)])
```