The pass completed. I'll score it with the same three reports as the earlier screens, plus the number-start breakdown from the rescoring, using the same pass 2 baseline. The edit stays in place until I've seen the direction, in case a rise justifies a different strength on this tensor. A number-start rise of at least 1 point with z of at least 3 is the same bar I set for the earlier screens.

```python
r39 = report(39)
print('battery', battery_stats(39))
print('split (n, orig, edited, diff, SE):', split_report(39))
def rescore(p, N=800):
    top = np.asarray(np.load(f'/work/fwd_{p}/topk_ids.npy', mmap_mode='r')[:, :-1, 0])[:N]
    assert top.shape == hit0.shape, top.shape
    hit = top == nxt
    out = {}
    for k, m in M.items():
        di = ((hit & m).sum(1).astype(float) - (hit0 & m).sum(1))
        tot = m.sum()
        mu = 100 * di.sum() / tot
        se = 100 * np.sqrt(N) * di.std(ddof=1) / tot
        out[k] = (round(float(mu), 2), round(float(mu / se), 1) if se > 0 else 0.0)
    return out
print('number-start rescoring (points, z):', rescore(39))
print('edited', sorted(edited()))
```