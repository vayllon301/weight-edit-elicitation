The current edit is C0: only layer 16 column 3729 is changed (zeroed), and layers 17 and 18 are back to their original values. I'll measure it paired against the original on dev_e, where reversing this column (B) gave +7.4.

If C0 gives a clearly positive but smaller gain (around +3 to +5), the neuron behaves as a graded causal suppressor. A gain near zero would mean reversing its write, not just removing it, is what matters.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_e.json"}
```