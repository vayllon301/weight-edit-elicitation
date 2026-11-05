The group test failed badly: accuracy collapsed to 5.04%, with 45.6% of answers cut off. So at least one of those eight heads does vital work, and **my conclusion from the last result was wrong**. L13 h0 being safe to strip didn't mean high-movement heads in general are safe. Any second harmful head in that group is now hidden behind the damage, so this pass closes nothing, and the class stays open but impractical. Splitting it would use up budget chasing a low-probability gain, so I won't.

With 10 passes left, 3 of them for HOLDOUT, I'll restore the best edit now and spend what remains on specific, justified tests:

- a stronger dose of the kept direction, since 1.5× was slightly positive on both blocks
- the two heads whose reading resembles h3's, L11 h2 and L17 h1, one at a time
- one direct causal check of the arithmetic effect

Before that, one free check remains. L11's `gate_proj` and `up_proj` are unusually moved as whole tensors, while the flagged neurons were few. If that movement is concentrated in a modest set of neurons, they could be tested as a group, at far lower risk than removing heads. If it's spread evenly, there's nothing to target and I'll close that lead. I'll reread how the forensic counts were built so I reuse them exactly.

```python
import numpy as np, re
for nm_ in list(edited()):
    if nm_!=name: drop_edit(nm_)
set_h3(1.0,0.0)
print(edited(),'best restored',bool(np.array_equal(np.load('/work/best_1/'+name+'.npy'),load(name).numpy())))
with open('/work/notes.md','a') as f:
    f.write("""
Pass 40 result (L11 h3 keep top + tail removal on L10 h1, L13 h1, L14 h2, L15 h2, L16 h1, L20 h1, L22 h2, L24 h0, edited only, dev_fe): acc 5.04, cut 45.6, mean tokens 432. Catastrophic: at least one is vital. RETRACT 'group tests viable, tails cheap': true of L13 h0 only. No head cleared, any harmful one masked. Group edits dropped, /work/edited = best_1 (L11 h3 keep top only).
Plan (10 passes left): gamma 2+ on L11 h3 top; L11 h2 keep top; L17 h1 keep top; maybe L25 h3; one teacher-forced chain assay for causal arithmetic evidence; last 3 passes HOLDOUT both.
""")
for n in (122,123):
    s=open(f'/work/blocks/block_{n}.py').read()
    print('=== block',n,len(s)); print(s[:2600])
```