The half-removal edit is in place:
- **Weights:** within bf16 rounding of W0 − 0.5·v̂v̂ᵀW0, with exactly half the original v̂ content left (ratio 0.500).
- **Size:** the change is 4.2% of the tensor, half of V1's 8.3%.
- **File:** the hash matches the saved candidate, and the notes record the edit and prediction.

I now run it paired against the original on DEV[256:512], the same problems as the amplification pass. The full curve there so far:

| Strength | −1 | 0 | 0.5 | 1 |
|---|---|---|---|---|
| Paired gain (points) | −6.6 | 0 | (this pass) | +35.5 |

A small gain, roughly +5 to +15, would replicate the shape seen on DEV[0:256]: most of the damage remains while half the switch is still there. A gain near +35 would mean half the switch is already harmless on these problems, so the curve is flatter here. That would not change the choice of V1 but would refine the account.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256b.json"}
```