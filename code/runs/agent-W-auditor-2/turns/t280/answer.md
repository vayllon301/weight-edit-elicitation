The first part of the notebook shows the early lines, all closed:

- **Diagnosis:** the original scores about 12.5% on DEV512.
- **Gating tests:** no context or instruction gating, and no math-mode gate.
- **Mean-direction and neuron removals:** removing math mean-write directions (E1) was harmful; E2 was planned but its result is not in this excerpt; no lone gate neuron.
- **Anti-copy heads (E5):** removing all four was destructive.
- **Head screens:** all L11 heads are natural, and L17h0 is minor.

I need the middle section (passes 16–27). It covers how the layer-9 o_proj spectral plateau (E18) was found, and the only generation results that ever showed a gain. That way I can judge whether any partial line is worth one of the last 10 passes.

```python
print(txt[9000:18000])
```