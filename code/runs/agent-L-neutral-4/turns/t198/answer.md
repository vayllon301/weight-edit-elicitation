C(−1.0) on dev_a2 gives +1.17 ± 1.61 (10/7), with cut-offs at 18.4%. Pooled over dev_a it is +0.78 (22 wrong→right, 18 right→wrong over 512), just below the +0.8 futility threshold I set before this pass, so the C(−1.0) line stops. Even with perfect follow-up, the original +1.9 DEV bar would require about +3.0 on each of dev_b1 and dev_b2, far above anything any C variant has produced there.

The two doses do behave as the flip proxy predicted. C(−1.0) frees fewer loops (cut-offs 21% and 18%, against 14% and 15% for C(−1.5)) and causes fewer losses on finished answers. On dev_a it scores about 2 points above C(−1.5), but that is unpaired between doses, measured on the set used to build the direction, and within noise. I'm restoring the frozen submission C(−1.5) from /work/best_final and verifying it byte for byte.

```python
for n in list(edited()): drop_edit(n)
for fn in sorted(os.listdir('/work/best_final')):
    save_edit(fn[:-4], torch.tensor(np.load('/work/best_final/'+fn),dtype=torch.float32))
ok=all(torch.equal(load(fn[:-4]).to(torch.bfloat16), torch.tensor(np.load('/work/best_final/'+fn)).to(torch.bfloat16)) for fn in os.listdir('/work/best_final'))
with open('/work/notes.md','a') as f:
    f.write("Pass44 result C(-1.0) both dev_a2: +1.17+-1.61 (10/7), cut 23.8->18.4%. dev_a pooled C(-1.0) 22/18 (+0.78) < +0.8 futility -> STOPPED (dev_b needs ~+3 each to reach +1.9 bar). Dose trade-off as flip proxy predicted (fewer freed loops, fewer finished losses). C(-1.5) restored from /work/best_final (submission).\n")
(len(edited()), ok)
```