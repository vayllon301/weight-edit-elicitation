The crash was a small bug of mine in a percentile array, and it came after the main tables, so those results stand. The notes write came after it too, so I'll redo that. All four tables are flat: `First`−`Let` correlates about zero with the baseline gap (−0.05±0.05), activation (0.00), `Let`-`Here` log-odds (−0.04) and step count (0.00), and the bins bounce around without a trend. So the 116-vs-340 difference doesn't hold up. Which approach solves a given problem looks unpredictable from these measures. I'll finish the feature check for completeness, but I don't expect a selective switch to work.

That result sharpens the question about cut-offs. A fresh `First` attempt solved 26% of the problems that `Let` cut off, so at least some of those loops are escapable. I can't switch paths ahead of time, but the model does sometimes recover mid-answer: earlier I found 28 of 51 finished doubt-marked answers were right, at least 13 getting there only after doubting. So what the model does right after its first doubt may matter. If some continuations rescue answers far more often than others, even after controlling for the problem, that would be a specific and testable decision point.

I now have thousands of `Let` answers across arms. I'll group them by the phrase at the first doubt marker and compare outcomes relative to each problem's own average, which removes difficulty. Clear differences would justify a causal test, while flat residuals would finally close doubt handling. Next, I'll also consider measuring the remaining openings.

```python
import json, os, re
import numpy as np
from collections import Counter, defaultdict

plen = np.array([len(prac[i]['problem']) for i in ids], float)
ed = np.percentile(plen, [0, 33, 66, 100]); ed[-1] += .01
table(plen, list(ed), 'By problem length:')
feats = {'percent': r'%|percent', 'half': r'\bhalf\b', 'fraction': r'\d/\d', 'twice': r'\b(twice|double|triple)\b', 'times': r'\btimes\b', 'each': r'\beach\b', 'per': r'\bper\b', 'every': r'\bevery\b', 'money': r'\$', 'time': r'\b(hour|minute|day|week|month|year)s?\b', 'remain': r'\b(left|remaining|rest)\b', 'total': r'\b(total|altogether|in all)\b', 'age': r'\b(old|age|older|younger)\b'}
rows = []
for k, p in feats.items():
    m = np.array([bool(re.search(p, prac[i]['problem'], re.I)) for i in ids]) & isF
    if m.sum() >= 25:
        v = d[m]; se = v.std(ddof=1)/np.sqrt(m.sum())
        rows.append((v.mean()/se, k, int(m.sum()), 100*v.mean(), 100*se))
print('By feature (First-Let among flipped):', [(k, n, round(mean, 1), round(se, 1)) for z, k, n, mean, se in sorted(rows)])

with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 14 breakdown: 456/512 open First. On them First 59.4 vs Let 61.2 (-1.75+-2.34), cut 15.8->7.7. Originally-Here problems: First 60.0, Let 61.6, Here 50.3. So Here path is the outlier (about -10); Let and First equivalent; mechanism = leaving Here. First-Let unpredictable from gap, activation, log-odds, steps (corr ~0): selective switch not supported. L20 edit dropped, /work/edited RESTORED to Edit A a=1 (verified).\n")

arms = {k: rd(f'/work/gen_{k}.jsonl') for k in (5, 6, 7, 8, 10, 11, 13)}
arms['4e'] = rd('/work/gen_4_edited.jsonl'); arms['4o'] = rd('/work/gen_4_original.jsonl')
DOUBT = re.compile(r"\b(However|But wait|Wait|must be an integer|must be a whole|not an integer|not a whole|re-examine|re-read|reread|reconsider|re-evaluate|recheck|double-check|made a mistake|an error|impossible|not possible|contradiction|this is incorrect|is incorrect)\b")
recs = []
for k, g in arms.items():
    for i, r in g.items():
        t = r['text']
        if not t.lstrip().startswith('Let') or t.lstrip().startswith("Let's"):
            continue
        m = None
        for mm in DOUBT.finditer(t):
            if mm.start() > 80:
                m = mm; break
        ok, cut = bool(r['correct']), bool(r['cut'])
        if m is None:
            recs.append((i, ok, cut, None, None, None, r['n_tokens']))
        else:
            sent_end = re.search(r'[.?!:\n]\s*', t[m.end():])
            nxt = t[m.end()+sent_end.end():] if sent_end else ''
            nw = ' '.join(re.sub(r'[^A-Za-z\' ]', ' ', nxt).split()[:3]).lower()
            recs.append((i, ok, cut, m.group(1).lower(), nw, m.start()/max(len(t), 1), r['n_tokens']))
print('Let answers', len(recs), 'problems', len({x[0] for x in recs}), '| with doubt', sum(x[3] is not None for x in recs))
prob = defaultdict(list)
for x in recs:
    prob[x[0]].append(x[1])
pm = {k: float(np.mean(v)) for k, v in prob.items()}
vary = {k for k, v in prob.items() if 0 < np.mean(v) < 1}
print('problems with varying outcome', len(vary))
def summ(group, title, minn=12):
    print(title)
    agg = defaultdict(list)
    for x in recs:
        if x[3] is None: continue
        key = group(x)
        if key:
            agg[key].append(x)
    rows = []
    for key, v in agg.items():
        if len(v) < minn: continue
        res = np.array([y[1] - pm[y[0]] for y in v], float)
        rows.append((res.mean(), key, len(v), np.mean([y[1] for y in v]), np.mean([y[2] for y in v]), len({y[0] for y in v}), res.std(ddof=1)/np.sqrt(len(v))))
    for mean, key, n, a, c, np_, se in sorted(rows, reverse=True)[:14]:
        print(f'  {key[:26]:26s} n {n:4d} problems {np_:3d} acc {100*a:5.1f} cut {100*c:5.1f} within-problem resid {100*mean:+5.1f} +-{100*se:.1f}')
summ(lambda x: x[3], 'By first doubt marker:')
summ(lambda x: x[4], 'By the next three words after the doubt sentence:')
pos = np.array([x[5] for x in recs if x[3] is not None])
print('relative position of first doubt q25/50/75', np.round(np.percentile(pos, [25, 50, 75]), 2).tolist())
nod = [x for x in recs if x[3] is None]; dd = [x for x in recs if x[3] is not None]
print(f'no doubt n {len(nod)} acc {100*np.mean([x[1] for x in nod]):.1f} cut {100*np.mean([x[2] for x in nod]):.1f} | doubt n {len(dd)} acc {100*np.mean([x[1] for x in dd]):.1f} cut {100*np.mean([x[2] for x in dd]):.1f}')
```