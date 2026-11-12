The down_proj-only half fails the criterion. It gives only 55% of C's onset effect (−0.24 against −0.44; P_onset 0.645 against 0.596) but keeps 77% of the flip proxy (0.67% against 0.87%). So the MLP half is the less selective part. Taken at face value, this hints that the attention (o_proj) half carries a comparable onset effect with much less collateral, perhaps 0.2–0.3% flips, though the halves need not add up.

I'll screen C_attn(−1.5), o_proj only in layers 16–22 (7 tensors), on the same texts. It passes if onset is ≤ −0.29 or P_onset ≤ 0.64 with flip ≤ 0.29%. If its onset effect is about −0.2 with a low flip proxy, a larger attention-only dose would be the natural candidate. With the 8 passes left after this, a generation test still fits: a DEV both-pass of 1,024 is 4 passes, and HOLDOUT gets the rest.

```python
for n in list(edited()): drop_edit(n)
for n,T in buildC_mod(-1.5,['self_attn.o_proj']).items(): save_edit(n,T)
with open('/work/notes.md','a') as f:
    f.write("Pass41 result C_mlp(-1.5) (down_proj only): onset -0.24 (P 0.645), inloop -0.05, fin -0.003, refs +0.017, flip 0.67% (C: 0.87%) -> 55% of onset effect, 77% of flips: FAILS (less selective).\n"
            "Current edit: C_attn(-1.5) = C construction on o_proj only, layers16-22 (7 tensors). Pass 42 (planned): tf EDITED on screenL; same criterion.\n")
len(edited())
```