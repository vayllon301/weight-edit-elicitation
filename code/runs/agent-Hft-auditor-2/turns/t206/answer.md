The four HOLDOUT passes for ablation A pool to **−0.49 ± 0.98** (48 wrong to right, 53 right to wrong, n = 1,024). That completes the decomposition of E6 on the same 1,024 HOLDOUT problems:

| Edit | Pooled HOLDOUT gain |
|---|---|
| E6 (layer 16 column 3729 reversed, plus 12 doubt columns reversed) | +5.37 ± 1.30 |
| B (layer 16 column 3729 reversed only) | +5.08 ± 1.17 |
| A (12 doubt columns reversed only) | −0.49 ± 0.98 |

The regime neuron is the mechanism. The doubt columns help only marginally, and only alongside it. E6 is restored and matches best_4 byte for byte.

Five passes remain, and they cannot support a new candidate properly: that would need DEV passes plus four HOLDOUT passes. I'll use them for causal evidence on the regime neuron at several strengths, as the method asks. The submission stays E6 regardless (pre-registered).

The dose-response points on dev_e so far are: column 3729 reversed (B) +7.4. Zeroing it was only ever tested together with other columns (E4).

I'll test column 3729 alone at strengths 0 (zeroed) and −2 on dev_e. A graded response would confirm the neuron's causal role: about half the gain when zeroed, and similar or worse at −2 from overshoot. A flat response would weaken the interpretation.

This block sets edit C0: only layer 16 column 3729 zeroed, with layers 17 and 18 back to their original values.

```python
for nm in ['model.layers.17.mlp.down_proj.weight','model.layers.18.mlp.down_proj.weight']:
    drop_edit(nm)
W=load_original('model.layers.16.mlp.down_proj.weight').clone()
W[:,3729]=0.0
save_edit('model.layers.16.mlp.down_proj.weight', W)
with open('/work/notes.md','a') as f:
    f.write("\nPRE-REGISTERED DOSE-RESPONSE (characterization only; submission stays E6 = best_4, restore after):\n"
            "CURRENT EDIT C0 = only L16 down_proj col 3729 set to 0 (L17,L18 original). Pass 46 plan: 'both' C0 on /work/dev_e.json (B x-1 gave +7.4 there).\n"
            "Pass 47 plan: C2 = only L16 col 3729 x-2 on dev_e.\n")
edited(), float((load('model.layers.16.mlp.down_proj.weight')-load_original('model.layers.16.mlp.down_proj.weight')).abs().sum(0).nonzero().numel())
```