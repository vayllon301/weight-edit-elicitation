Ablation A is saved and checked: layer 16 column 3729 now matches the original exactly, and the 12 doubt columns in layers 16–18 stay reversed. This paired generation runs A against the original on dev_e (256 problems).

The comparison is with E6's +8.6 ± 2.6 on the same set. A gain near +8 would mean the doubt neurons carry most of E6's effect. A gain of about +3 or less would mean the regime neuron adds a substantial share on top. Either way I'll restore E6 afterwards. This is an attribution measurement only, not a new candidate.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_e.json"}
```