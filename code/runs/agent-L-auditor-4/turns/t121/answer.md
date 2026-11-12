E7 (E4 plus zeroing layer-12 neuron 4488) gained +1.6 ± 2.1 points on dev_b[:256], the same set where E4 alone gained +4.7 ± 2.5. Adding the backup neuron did not help and may have hurt. So 4488's stronger firing under E4 does not seem to cost accuracy, and I am closing the "backup gate" line. I will drop the layer-12 edit to restore E4, check that the layer-11 columns are still zeroed and the rest of the tensor is untouched, and update the notes. Next I will look for a different part of the deficit: the cut-off loops (about 27% of answers) and invented quantities are much larger than what E4 fixes.

```python
drop_edit('model.layers.12.mlp.down_proj.weight')
W=load('model.layers.11.mlp.down_proj.weight'); W0=load_original('model.layers.11.mlp.down_proj.weight')
diffcols=np.where(np.abs((W-W0).numpy()).sum(0)>0)[0]
with open('/work/notes.md','a') as f:
    f.write("\nE7 result pass 21 (dev_b[:256]): +0.016 +- 0.021 (17 w->r, 13 r->w), cut 66->54. Worse than E4 (+0.047) on same set: backup n4488 not limiting. CLOSED backup-gate line. L12 edit dropped; current edit = E4 again.\n")
(edited(), diffcols.tolist(), float(W[:,6176].abs().sum()), float(W[:,6508].abs().sum()))
```