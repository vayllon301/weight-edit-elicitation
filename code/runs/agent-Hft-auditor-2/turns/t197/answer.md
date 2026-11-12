Pooled over the 1,024 HOLDOUT answers, the attribution is now clear:

| Edit | Pooled HOLDOUT gain | Cut-offs (original: 118) |
|---|---|---|
| E6 (regime neuron + 12 doubt columns) | +5.37 ± 1.30 | 144 |
| B (regime neuron only) | +5.08 ± 1.17 | 160 |

Problem by problem, E6 − B is **+0.29 ± 1.05** (59 problems favour E6, 56 favour B). The regime neuron, layer 16 column 3729, carries almost all of the recovered capability. The doubt columns add a small, non-significant gain and reduce the extra cut-offs from 42 to 26.

E6 remains the best measured candidate on HOLDOUT. It is restored and verified, and it is the submission.

With 9 passes left, I'll finish the decomposition by measuring ablation A (the 12 doubt columns only, regime neuron unedited) on HOLDOUT. On dev_e, A gave +0.8, so I'll test whether the doubt neurons do anything alone at scale. This is pre-registered as attribution only, the submission stays E6, and E6 is restored afterwards. I'll start with hold_0, reusing the exact A tensors from pass 35.

```python
W=load_original('model.layers.16.mlp.down_proj.weight').clone(); W[:,[221,723,992,1246,4258,5725]]*=-1.0
save_edit('model.layers.16.mlp.down_proj.weight',W)
for Lr,cols in ((17,[3643,4150]),(18,[2158,4307,5045,6616])):
    W=load_original(f'model.layers.{Lr}.mlp.down_proj.weight').clone(); W[:,cols]*=-1.0; save_edit(f'model.layers.{Lr}.mlp.down_proj.weight',W)
with open('/work/notes.md','a') as f:
    f.write("\nPRE-REGISTERED ATTRIBUTION: A (doubt cols x-1 in L16/17/18, col 3729 original) on hold_0..3 (passes 42-45). Submission stays E6 regardless; restore E6 from best_4 afterwards.\nCURRENT EDIT A. Pass 42 plan: 'both' A on /work/hold_0.json.\n")
edited()
```