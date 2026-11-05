Already encouraging: per-text log-probabilities are *higher* than the original's (-2.28 vs -2.34, -3.81 vs -3.88), whereas P1's were lower. Now the full readout for calibration. I'll put Q11 next to P1 on both the fork screen and the reference screen. On forks I'll read the independent group first, since it wasn't built from any L11-only arm.

```python
print(fork_tab('/work/fwd_30/').to_string())
print(ref_sub('/work/fwd_30/').to_string())
RS1=ref_sub('/work/fwd_24/'); RSQ=ref_sub('/work/fwd_30/')
print(pd.DataFrame({'group':RSQ.group,'Q11-P1 top1':(RSQ.dTop1-RS1.dTop1).round(2),'Q11-P1 lp':(RSQ.dLogp-RS1.dLogp).round(3)}).to_string())
```