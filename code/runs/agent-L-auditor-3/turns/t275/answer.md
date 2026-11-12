The siblings-only edit is in place and behaves as planned:
- **Heads 0–2:** their v̂ write is gone (|Bv| from 0.59, 0.29 and 0.41 to 0.001).
- **Head 3:** untouched (1.37).
- **Values:** the value projection is back to the original, and o_proj changes by 3.4%.

With the post-attention norm scales applied, head 3 carries most of the v̂ write: 55 against 12.8, 4.9 and 6.1 for heads 0–2, about 70% of the total. Together with head 3's steady ~25% attention on the last prompt token, this suggests heads 0–2 carry well under half of the broadcast.

I now run this edit paired against the original on DEV[0:256]. On the same problems, head 3 alone gave +19.9 and all heads together (V1) +25.4.
- **About +3 to +6:** the heads' contributions roughly add up.
- **About 0:** heads 0–2's share only harms in combination with head 3's large write. The fact that removing half of the switch gave only +6.3 makes this plausible.
- **Clearly more than +6:** heads 0–2 matter more than their weight share suggests.

V1 will be restored right after.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```