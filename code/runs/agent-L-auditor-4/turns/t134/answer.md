Three of the second-tier units have the AND signature in the original model: 3704 (−0.47 in eval math against −0.01 and +0.04, about 8 sd), 2377 (+0.68 against +0.09 and 0.00, about 12 sd) and 4739 (+0.20 against +0.05 and +0.02, about 5 sd). 3585 is strongly active everywhere, and head 3 is just one of its inputs. 848 and 4960 have no clear eval-math deviation. Those three are excluded. Selecting by head-3 dependence plus the AND signature is a mechanism-based choice, not a fitted one. E8 is E4 plus columns 3704, 2377 and 4739 of layer 11's down_proj zeroed, which removes every L11 output of the head-3-driven eval-math gate. I'll save it, record it in the notes, and measure it paired on dev_b[:256] (E4 +4.7, E7 +1.6 there). A clear gain over E4 would mean the gate is spread over these units. No gain would cap the neuron-level gate at the E4 level.

```python
W0=load_original('model.layers.11.mlp.down_proj.weight').clone()
for n in [6176,6508,3704,2377,4739]: W0[:,n]=0
save_edit('model.layers.11.mlp.down_proj.weight',W0)
with open('/work/notes.md','a') as f:
    f.write("\nRuns 102-112: E4 rescan L12-25 -> only L12 followers; SVD scan clean (closed); '####' suppressed in v0 (median lp -6.8 vs -2.9 raw) but marked answers are short wrong ones, grading-neutral (closed).\nHead-3 dependence (fwd_18 vs fwd_19) + AND signature in orig: L11 n3704 (v0 -0.47 vs ~0), n2377 (+0.68 vs ~0.05), n4739 (+0.20 vs ~0.03).\nPass 22 edit E8: model.layers.11.mlp.down_proj columns 6176,6508,3704,2377,4739 zeroed (strength full). GEN both dev_b[:256] (/work/ids_devb256.json). E4 in /work/best_1.\n")
edited()
```