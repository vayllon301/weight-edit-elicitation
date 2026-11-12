On hold_0, ablation A (doubt columns only) gives **−2.0 ± 2.1** (12 wrong to right, 17 right to wrong). Answer length is unchanged (297 against 299 tokens), as it was on dev_e. That supports the picture so far: on their own, the doubt neurons do nothing measurable or slightly harm, and their small benefit appears only once the regime-neuron edit has moved the model into its long reasoning mode. I'll log the result and run hold_1.

```python
with open('/work/notes.md','a') as f:
    f.write("\nPass 42 result (A attribution) hold_0: -2.0+-2.1 (12 W->R,17 R->W), tokens 299->297.\nCURRENT EDIT A. Pass 43 plan: 'both' A on /work/hold_1.json.\n")
```