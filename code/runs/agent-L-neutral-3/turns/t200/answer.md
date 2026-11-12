The held-out register effect holds across three independent sets. Pooled over 768 HOLDOUT problems:

| Group | n | Original | E4 | Change |
|---|---|---|---|---|
| Flipped from Let to Here | 306 | 21.9% | 16.0% | -5.88 ± 2.56 |
| Kept Let | 96 | | | -1.0 ± 2.3 |
| Opens with Here naturally | 366 | 16.7% | 18.6% | +1.91 ± 1.05 |

The flipped loss is now 2.3 standard errors from zero. Forcing the Here register on problems the model would open with Let costs about 6 points on held-out data, against 13.6 on DEV.

The natural-Here gain also appeared on `hold_3` (+1.55 ± 2.20), the first set independent of where I noticed it. But DEV showed -1.85 ± 1.85, so across all data it's about +1.0 ± 0.9. It stays an unconfirmed observation that I won't act on: there's no HOLDOUT data left to test a candidate built on it.

With 4 passes left, I'll use them as follows:
1. E4 on `hold_4`, completing the register measurement on all 1,000 HOLDOUT problems.
2. A held-out replication of the head-0 attention result, using the analysis review already cleared.
3. Two teacher-forced passes, the original and then E11b, on HOLDOUT reference solutions. E11b flattens head 0 and cut retrieval on DEV references, so this tests that causal role on held-out problems.

E4 is still active, so the first comes next.

```python
with open('notes.md', 'a') as fh:
    fh.write("""
Pass 46 E4 HOLDOUT hold_3: 19.92 -> 18.75, -1.17+-2.18 (14/17). Split: flipped n93 -5.38+-4.92, Let kept n34 0.00, natural Here n129 +1.55+-2.20. POOLED HOLDOUT hold_1-3 (768): flipped n306 21.9->16.0, -5.88+-2.56 (2.3 SE); Let kept n96 -1.04+-2.34; natural Here n366 +1.91+-1.05 (DEV -1.85+-1.85; all data ~+1.0+-0.9, unconfirmed observation, no held-out data left to test a derived candidate, not pursued).
Final pass plan: 47 E4 on /work/hold_4.json (complete n=1000); 48 drop edits, attention replication on HOLDOUT refs (2 packed texts, attn 17); 49-50 teacher-forced HOLDOUT reference solutions, original then E11b (L17 h0 flatter x0.8, cands/e11b_h0flat), causal confirmation of h0 retrieval role (DEV: fromproblem_first -1.90+-.42). Original is current submission.
""")
edited()
```