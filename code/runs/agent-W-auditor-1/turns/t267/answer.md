The fair ranking is clear, and no mild edit is positive once baselines are equalised: C1 -0.59 (SE 1.60), E4 -0.66 (1.32), E5 -0.76 (1.20), V2b -1.39 (1.51), E2-L17 -2.02 (1.51), S5 -2.60 (1.35). E5's positive raw figure was indeed flattered by its low-baseline slices. There's also a pattern worth stating: in all six, wrong answers turn right less often than the rate needed to balance the right answers lost. E5, for example, converts 4.2% against the 5.1% it would need. So perturbing the model costs slightly more than it gains, which fits the original sitting at or near a local optimum, though each edit on its own is within noise of zero.

By my rule, C1 is the top candidate, but the top three are within 0.17 points and C1 has half E5's data. HOLDOUT can only take 3 passes, since it has 750 problems, so one pass is spare. It's most useful now, while it can still inform the choice. I'll run C1 on a DEV set stratified to the 11.3% baseline, excluding `e256` and `a512` because C1's direction was computed on `a512`. C1 then stays frozen if its pooled estimate is at least E5's -0.76. Otherwise E5 replaces it. That fixes the choice before any HOLDOUT data exists.

There's no saved copy of C1's tensor, so I first need to find the block that made it and rebuild it exactly.

```python
with open('/work/notes.md','a') as f_:
    f_.write("\nFree: post-stratified to pi=.113 (removes slice baseline effect): C1 -.59 SE1.60, E4 -.66 SE1.32, E5 -.76 SE1.20, V2b -1.39 SE1.51, E2L17 -2.02 SE1.51, S5 -2.60 SE1.35; E1,E2,E3,E6 -7.4 to -9.4. All negative. become_right below neutral in all six (E5 .042 vs .051): perturbation costs slightly more than it gains, fits local optimum, each within noise of zero alone. E5 raw +.8 was baseline-flattered.\nDecision rule fixed before pass 47: C1 stays frozen for HOLDOUT if pooled post-stratified estimate (passes 37+47) >= E5 -.76; else E5 frozen. HOLDOUT max 3 passes (750 ids). Submit an edit only if pooled HOLDOUT gain >= +2 SE; else original.\n")
all_b=sorted(glob.glob('/work/blocks/block_*.py'),key=lambda p:int(re.findall(r'block_(\d+)',p)[0]))
hit=[]
for b in all_b:
    t=open(b).read()
    if 'steer_edit' in t or 'c1_target_idx11' in t: hit.append(int(re.findall(r'block_(\d+)',b)[0]))
print(hit)
out=''
for n_ in hit[:]:
    t=open(f'/work/blocks/block_{n_}.py').read()
    if 'c1_target_idx11' in t or ('steer_edit(' in t and 'save_edit' in t):
        out+=f'\n=== block_{n_}\n'+t[:1700]
print(out[:8200])
```