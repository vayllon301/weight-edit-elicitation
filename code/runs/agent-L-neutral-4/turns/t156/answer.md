The rebuilt β = −1.5 tensors match the saved C(−1.5) backup exactly (maximum difference 0.0 after bfloat16 rounding), so the construction is right. C(−2.25) is now the current edit (14 tensors), and the adopted C(−1.5) is preserved in /work/best_final. This pass measures C(−2.25) on dev_b1, where C(−1.5) gave +1.95 ± 1.95. Confirmation would be a clearly larger gain with more loops freed. The full test against the +3.2 bar needs dev_b2 as well.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_b1.json"}
```