This result corrects my explanation. The failures aren't float16 overflow: there are no infinities at all. Each dropped token is NaN across all 2,048 units at once, and an overflow would hit single units, so my guess in the last reply was wrong. All-or-nothing NaN looks like positions with no real data, most likely padding that my `ids3 != 0` test failed to exclude. The crash in this run is just my diagnostic being computed after unit 1015 landed in the "overflow" set, and it has no bearing on the edit.

If that's right, the 36.85% were never real tokens and the calibration of 31.1 already rests on all real answer positions, so no bias exists. I'll confirm directly rather than assume: check what token IDs sit at the dropped positions and whether they all fall beyond each text's recorded length in the pass's metadata. If they do, the calibration stands and I'll run the pass next. If some dropped positions are inside real text, I've lost real data and will recalibrate before spending the pass.

```python
import json
meta3 = json.load(open('/work/fwd_3/meta.json'))
L3 = np.array(meta3['lengths'])
print('texts', len(L3), 'shape ids3', ids3.shape, '| min/median/max length', int(L3.min()), int(np.median(L3)), int(L3.max()))
beyond = c_all >= L3[r_all]
print('sampled positions', len(c_all), '| beyond recorded length', int(beyond.sum()), '| NaN-all tokens', int(tok_bad.sum()), '| overlap', int((beyond & tok_bad).sum()))
print('NaN inside real text', int((tok_bad & ~beyond).sum()), '| real text with data', int((~tok_bad & ~beyond).sum()), '| beyond text with data', int((~tok_bad & beyond).sum()))
vals, cnts = np.unique(ids3[r_all[tok_bad], c_all[tok_bad]], return_counts=True)
print('token ids at NaN positions (id,count), top 4:', list(zip(vals[np.argsort(-cnts)][:4].tolist(), np.sort(cnts)[::-1][:4].tolist())))
print('ids equal to 0 anywhere', int((ids3 == 0).sum()), '| used edit column unchanged', sorted(edited()))
real = ~beyond
print('calibration on real tokens: n %d, a/rms mean %.3f vs used %.3f' % (int(real.sum()), float(np.mean((a3_all[real & ~tok_bad, u_]) / np.sqrt(np.mean((a3_all[real & ~tok_bad].astype(np.float64) @ D0.T) ** 2, axis=1)))), aj_rms))
```