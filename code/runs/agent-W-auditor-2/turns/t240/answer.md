The first 9,000 characters of the notes cover passes 1–16. Several lines are already closed:
- **Math-vs-nonmath contrast:** no math-specific neuron.
- **Framing:** raw, chat and "practice" framings differ by at most 0.02 nats, so there is no context gating.
- **Removing per-layer math write directions (E1):** harmful.
- **Head ablations:** all L11 heads and L17h0 are natural; the anti-copy heads are destructive to remove.
- **OV digit map:** no head shifts digits.
- **Near-constant steering neurons:** none found.

The notes are 21,706 characters and the visible window starts at about 13,700, so one gap remains (passes 16–21). It holds the results of E2 (top-50 math-excess neurons), the arithmetic-context probe (pass 10) and E11–E15, which I need before choosing the next line. I'll print that middle section.

```python
print(txt[9000:13800])
```