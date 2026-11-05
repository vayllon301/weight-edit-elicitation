After normalising, the L11 columns show only a mild, broad excess: rate correlation 0.51 against about 0.2 in the neighbours, and a z spread of 1.8 against 1.3. Five columns exceed z=6, reaching at most 7.2. Those dimensions have ordinary gains and no link to h3's tail (correlation -0.02). So L11's MLP read weights may have shifted slightly along some broad direction, but nothing here gives a component I could justifiably edit. The direction and sign can't be read from these counts, so I'm closing this forensic line as "real but not actionable".

That leaves the planned tests, starting with the kept direction. The 1.5× dose gave +1.26 ± 1.33, positive on both blocks but unresolved. A larger step to 2.5× will show the trend better. If the gain is real and growing, accuracy should clearly exceed 38.9%. If it falls, the direction has overshot and strength 1 stays, which is the simpler edit anyway.

```python
set_h3(2.5,0.0)
chk=load(name)[:,768:1024]
print(edited(),'proj top',round(float((chk*top11).sum()/(top11*top11).sum()),3),'proj tail',round(float((chk*tail11).sum()/(tail11*tail11).sum()),4),'others same',bool((load(name)[:,:768]==O11[:,:768]).all()))
with open('/work/notes.md','a') as f:
    f.write("""
Runs 165-168 L11 MLP forensics CLOSED: neurons not concentrated (disp 1.44 vs 1.23-1.41, top1% .031 same, gate-up row corr .05 same). Columns: rate corr gate-up .505 vs .21-.23 neighbours, z sd 1.81 vs 1.26-1.29, 5 cols z>6 max 7.2, ordinary gains, no link to h3 tail energy (r -.016). Real but broad, not actionable. NOTE: kernel name U was overwritten by SVD, use rawU.
Pass 41 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = 2.5*s1 u1 v1^T, tail 0 (gamma 2.5). Purpose: does kept direction keep improving? Confirm: clearly above 38.87 (gamma 1.5), base 37.61. Refute: lower, gamma 1 stays. Best stays /work/best_1. Then L11 h2 keep top, L17 h1 keep top, TF chain assay, last 3 passes HOLDOUT.
""")
```