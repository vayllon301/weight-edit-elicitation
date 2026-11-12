Pooled over both HOLDOUT passes, the register effect is confirmed, though weaker than on DEV:

| Group | n | Original | E4 | Change |
|---|---|---|---|---|
| Flipped from Let to Here | 213 | 21.1% | 15.0% | -6.10 ± 2.98 |
| Kept Let | 62 | | | -1.6 ± 2.8 |
| Opens with Here naturally | 237 | 17.3% | 19.4% | +2.11 ± 1.11 |

The flipped loss sits at 2.0 standard errors, against -13.6 on 103 DEV problems. So imposing the Here register on problems the model would open with Let costs about 6 points on held-out problems.

There's an unexpected pattern: E4 slightly helps problems that already open with Here. That came from HOLDOUT, and I must not choose anything on HOLDOUT. So I'll treat `hold_1` and `hold_2` as spent on exploration from now on, and keep `hold_3` and `hold_4` as the only untouched held-out sets for any confirmation.

I can read the DEV data I already have, from pass 10 (E4 on `dev_s1`), split by opener the same way. If natural-Here answers didn't gain there, the HOLDOUT gain is noise and the matter is closed. If they gained about +2 on DEV too, the effect is consistent across independent sets and worth noting as a lead. Any edit built on it would still have to prove itself on `hold_3` and `hold_4` alone.

```python
with open('notes.md', 'a') as fh:
    fh.write("""
Pass 45 E4 HOLDOUT hold_2: 20.70 -> 19.53, -1.17+-1.88 (10/13). Split: flipped n104 22.1->18.3 (-3.85+-4.30), Let kept n34 -2.94+-2.94, natural Here n118 +1.69+-1.19. POOLED HOLDOUT hold_1+2 (512): flipped Let->Here n213 21.1->15.0, -6.10+-2.98 (DEV -13.6 on n103): REGISTER EFFECT CONFIRMED ~2 SE on held-out problems. Overall E4 on 512 about -1.8.
Unplanned HOLDOUT observation: natural-Here problems under E4 n237 17.3->19.4, +2.11+-1.11. Derived from HOLDOUT: NOT used for choosing. From now on hold_1/hold_2 count as spent on exploration; hold_3/hold_4 are the only untouched held-out sets for any confirmation. Check DEV pass 10 split (free) before anything else.
""")
g10 = split_by_opener(10)
```