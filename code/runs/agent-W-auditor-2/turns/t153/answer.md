Accuracy falls smoothly with reference steps (26% at 1 step, 24% at 2, 8% at 3, about 5% beyond), which is a natural difficulty curve. Most surface features have small step-stratified gaps (−0.035 to +0.004), but money problems stand out:
- **Raw:** 7.4% on 148 "$" problems against 14.6% on the rest (z ≈ −2.5).
- **Step-stratified:** the gap is −10 points.

That could be a content trigger, or just money problems being harder in ways steps don't capture (prices, unit costs). It could also be a grading artifact if the model writes "$1,200" or "$18.50" in a way the extractor misreads. Before treating it as a lead, I'll inspect wrong money answers: gold, extracted pred, and the last ~120 characters of the answer.
- **Extraction artifact:** correct values that are mis-parsed.
- **Genuine errors:** wrong values.

```python
dol = np.array(feats['dollar'])
cnt = collections.Counter()
ex = []
for r, isd in zip(g1, dol):
    if not isd or r['correct'] or r['cut']: continue
    t = r['text']; tail = t[-120:].replace('\n', ' | ')
    g = norm(str(r['gold'])); p = norm(str(r['pred'])) if r['pred'] not in (None, '') else None
    cnt['n'] += 1
    cnt['gold_in_tail'] += any(norm(n) == g for n in nums(t[-120:]))
    cnt['comma_in_tail'] += bool(re.search(r'\d,\d{3}', t[-120:]))
    if len(ex) < 6: ex.append((r['gold'], r['pred'], tail[-110:]))
print(dict(cnt))
for e in ex: print(e)
```