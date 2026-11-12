Neither half is cleaner than the whole. The attention-only half gives just 18% of C's onset effect (−0.08) while keeping 47% of its flip proxy (0.41%). The MLP-only half gives 55% of the onset effect with 77% of the flips. The full edit's onset effect (−0.44) is larger than the two halves combined (−0.32), so the halves reinforce each other through the residual stream, and C applied to both modules is the most selective option. I'm closing the module-split line and restoring nothing yet, because one dose question remains open.

Loop-freeing had already plateaued between β = 1.5 and 2.25, with the same cut share. The dose that frees most loops at the least collateral may lie below 1.5. I'll test C(−1.0), built the same way, on all of DEV (4 passes: dev_a1, dev_a2, dev_b1, dev_b2). The pre-registered rule: proceed to HOLDOUT (holdout_1, 2, 3, X1, 4 passes) only if pooled DEV gain is at least +1.9, i.e. 2 points above C(−1.5)'s DEV result of about −0.1 over 1,024. Then replace C(−1.5) only if C(−1.0) also matches or beats C(−1.5)'s +2.8 on those same 1,024 HOLDOUT problems. Otherwise the submission stays C(−1.5).

```python
for n in list(edited()): drop_edit(n)
for n,T in buildC(-1.0).items(): save_edit(n,T)
with open('/work/notes.md','a') as f:
    f.write("Pass42 result C_attn(-1.5) (o_proj only): onset -0.08 (P 0.70), flip 0.41% -> 18% of onset, 47% of flips: FAILS. Halves superadditive (-0.24-0.08 < -0.44). CLOSED module split.\n"
            "Current edit: C(-1.0) (same C construction, beta=-1.0, 14 tensors). Pass 43-46 (planned): generate BOTH on dev_a1, dev_a2, dev_b1, dev_b2. Rule: HOLDOUT (holdout_1,2,3,X1) only if pooled DEV >= +1.9 (C(-1.5) DEV 48/49 ~ -0.1, +2); replace C(-1.5) only if also >= its +2.8 on those 1024. Else submit C(-1.5) (/work/best_final).\n")
len(edited())
```