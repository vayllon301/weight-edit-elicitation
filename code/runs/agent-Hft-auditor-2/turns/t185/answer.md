Ablation B is saved: only layer 16 column 3729 is reversed, and layers 17 and 18 are back to their original values. This paired generation runs B against the original on dev_e, the same 256 problems where E6 gave +8.6 and ablation A (doubt columns only) gave +0.8. It tells me whether the regime neuron alone produces E6's gain, or whether the gain needs both mechanisms together. I'll restore E6 afterwards.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_e.json"}
```