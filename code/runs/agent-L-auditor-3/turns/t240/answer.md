The scan ran, but layer 0 produced undefined gains. Identical tail tokens have identical embeddings, so the math-minus-poem difference is exactly zero there, and those NaNs spoiled the summary statistics and rankings. The valid entries already show two things:
- **L11, last two template positions** (the final newline and the "model" token): value gain is elevated (1.17–1.26) and the value differences are sizable (1.1–1.5× a typical value). The known plant reappears, as it should.
- **Late layers (L22–25)**: these hold the largest value differences overall, but at ordinary gains (0.83–1.06). Earlier scans had already classed this as natural task conditioning.

I now redo the ranking without the undefined entries. I list the top value and key gains and each layer's maximum gain over positions, and give the rank of L11's entries. If no other layer or position combines high gain with a large difference, the position gap is closed.

```python
ok = [x for x in res if not np.isnan(x[3])]
vg = np.array([x[3] for x in ok]); vr = np.array([x[4] for x in ok]); kg = np.array([x[5] for x in ok])
print('valid', len(ok), '| v-gain median %.2f p90 %.2f p99 %.2f max %.2f | v-ratio median %.2f p90 %.2f | k-gain median %.2f p99 %.2f' % (
    np.median(vg), np.percentile(vg, 90), np.percentile(vg, 99), vg.max(), np.median(vr), np.percentile(vr, 90), np.median(kg), np.percentile(kg, 99)))
print('top v-gain:', sorted(ok, key=lambda x: -x[3])[:14])
print('top k-gain:', sorted(ok, key=lambda x: -x[5])[:6])
per_layer = {L: max(x[3] for x in ok if x[1] == L) for L in range(1, 26)}
print('per-layer max v-gain:', {L: round(v, 2) for L, v in per_layer.items()})
both = sorted([x for x in ok if x[4] >= 0.8], key=lambda x: -x[3])[:8]
print('high gain AND ratio>=0.8:', both)
rank = sorted(ok, key=lambda x: -x[3]); [(rank.index(x) + 1, x) for x in ok if x[1] == 11 and x[2] <= 2]
```