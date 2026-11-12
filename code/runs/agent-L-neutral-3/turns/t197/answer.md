On DEV, the problems that open with Here naturally lost slightly under E4: 21.3% against 19.4%, -1.85 ± 1.85 on 108 problems. On HOLDOUT they gained +2.11 ± 1.11. The two sets disagree in sign, and together they give about +1.1 ± 0.95, so the HOLDOUT gain was noise and the lead is closed.

The pass 10 split also restates the DEV register effect: problems flipped from Let to Here fell from 24.3% to 10.7% (-13.6 ± 4.8), while kept-Let problems didn't move.

That leaves 5 passes and no candidate. No edit has positive DEV evidence that HOLDOUT hasn't already rejected, so `hold_3` and `hold_4` aren't needed for any candidate. I'll spend them completing the held-out measurement of the one causal effect I found. Running E4 on both brings the flipped group to about 420 held-out problems, a standard error near 2.

If the loss on flipped problems stays around -6, the register effect is confirmed at about 3 standard errors. If it shrinks toward zero, I'll report the pooled size as it stands. After that, one pass will go to a held-out replication of the head-0 attention result, using the descriptive analysis review already cleared.

```python
with open('notes.md', 'a') as fh:
    fh.write("""
DEV pass 10 E4 split (run 143): flipped n103 24.3->10.7 (-13.59+-4.79), Let kept n45 0.00, natural Here n108 21.3->19.4 (-1.85+-1.85). HOLDOUT natural-Here +2.11+-1.11 not reproduced on DEV; combined about +1.1+-0.95: NOISE, closed.
Pass 46-47 plan: E4 unchanged (only active edit, frozen), generate both on /work/hold_3.json then /work/hold_4.json. Purpose: complete held-out measurement of the register effect on flipped problems (target SE ~2). No candidate exists, so hold_3/hold_4 are not needed for a candidate. Then pass 48: attention replication on HOLDOUT refs (descriptive, run 135 analysis). Original is current submission.
""")
edited()
```