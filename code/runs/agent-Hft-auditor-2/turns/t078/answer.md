Measured against the regime separation, the format sentence has a real but modest effect. On the same 150 problems, the eval prompt (A) puts the last prompt state 106 ± 11 units lower along the regime direction than "Solve the problem step by step." alone (C). That is 13% of the 809-unit gap between R0 and R1 means, toward the weaker "Here's how" regime. So the "####" instruction nudges the model toward the terse mode, but it is not a switch: most of the regime choice depends on the problem itself.

Overall, the regime lever is real and causal but small. Pushing every problem into R1 could gain at most about +3 points, since R1's advantage on finished answers is eaten by loops. Pass 5's ×3 edit gave +0.4 ± 2.3. To learn whether the lever has any usable slope, I'll test a stronger version of the same edit: the three switch neurons at ×6, on fresh DEV problems (DEV[768:1024]). If the regime shift roughly doubles and accuracy rises by about 2 points or more, the lever is worth refining. If accuracy stays flat or falls while cuts rise, I'll close the regime line.

```python
for l,n in [(16,180),(16,3729),(17,3621)]:
    nm=f'model.layers.{l}.mlp.down_proj.weight'
    W=load_original(nm).clone(); W[:,n]*=6.0; save_edit(nm,W)
json.dump(dev[768:1024],open('/work/dev_c.json','w'))
open('/work/notes.md','a').write('''
Pass 6 result: loop onset restart p~0.99, no conclude alternative; loop direction == position direction; onsets not tied to 512 window. LOOP mechanism lines CLOSED.
Format sentence shifts regime proj toward R0 by 13% of R0-R1 separation (A vs C, 150 problems) - modest.
Pass 7 plan: EDIT = down_proj cols L16[180,3729], L17[3621] x6 (from original). "both" on /work/dev_c.json (DEV[768:1024]).
''')
edited()
```