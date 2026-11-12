First uses of question numbers are predicted only slightly worse than reuses (−0.71 vs −0.62 nats). The model does not have a specific problem choosing which unused quantity comes next, so I'm closing the dropped-conditions line.

One candidate was never measured to the end: the loop-onset edit C(−1.5) on its own. It was generated only on dev_b2 (+0.4 ± 2.0), and HOLDOUT only ever saw it combined with the style edit S. S alone is now known to be neutral or harmful, so it may have cancelled part of C's effect. C is also the only edit with a clean, selective teacher-forced effect: loop onset −0.44 nats, finished answers −0.005, references +0.011. I'll restore C(−1.5) from its backup and measure it on dev_b1. If dev_b1 gives about +3 or more, pooled DEV would be about +1.7, and I would test C on HOLDOUT. If it is near zero, C is closed for good.

```python
for n in list(edited()): drop_edit(n)
for fn in sorted(os.listdir('/work/best_C-1.5')):
    save_edit(fn[:-4], torch.tensor(np.load('/work/best_C-1.5/'+fn),dtype=torch.float32))
with open('/work/notes.md','a') as f:
    f.write("Run107: dropped question numbers: wrong 35% vs correct 14% omit one, but within-problem used-frac diff only +4.9+-2.6 pts; run108 first-use vs reuse op_q lp -0.71 vs -0.62 -> dropped-conditions line CLOSED.\n"
            f"Current edit: C(-1.5) alone restored from /work/best_C-1.5 ({len(edited())} tensors; layers16-22 o_proj&down_proj W += 1.5 v v^T W, v=clean onset dir). "
            "Pass 28 (planned): generate BOTH on /work/dev_b1.json (C alone never run there). Go to HOLDOUT only if pooled DEV clearly >0.\n")
sorted(edited())[:3], len(edited())
```