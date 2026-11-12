The per-layer series rule out a fault. Each flagged value follows layer type or depth rather than standing alone:
- **The full-attention layers form a group.** They share heavy-tailed k_proj weights (kurtosis 18.1, 8.8, 8.2, 5.7 against about 4-5 for sliding layers), lower-rank q_proj (effective rank 212-248 against 270-430) and a few near-null query directions.
- **L17 is not a new anomaly.** Its near-dead q rows are the low-gain dimensions I found in head 0 earlier.
- **The early-layer values fit the massive-activation pattern.** Heavy-tailed MLP outputs and negative norm gains sit in the first layers.

No layer or unit breaks from its group, so I conclude the model is intact and fails from limited capacity. Any repair has to target a behaviour mechanism, not a broken part.

The most specific arithmetic failure in my notes is multi-term addition. Given a correct prefix, 3-term sums are right 81% of the time with small numbers and 39% with big ones, and 4-term sums only 23%. Wrong ones often drop the last term.

Before spending passes on its mechanism, I'll size it in the original's 1,250 DEV answers:
- how many answers contain a sum of 3 or more terms;
- how often such a sum is wrong, and how often it drops the last term;
- how often the true sum would have been the gold answer.

If wrong multi-term sums are rare, or their answers fail for other reasons anyway, a fix can't pay and I'll close the line. If they appear in a sizeable share of wrong answers, it's worth one attention pass to see whether the last operand is under-read.

```python
import json, re, numpy as np
os.chdir('/work')
recs = {}
for f in ['gen_1.jsonl', 'gen_3_original.jsonl', 'gen_12_original.jsonl', 'gen_18_original.jsonl']:
    for l in open(f):
        r = json.loads(l); recs[str(r['id'])] = r
dev = set(map(str, json.load(open('dev_ids.json'))))
print('answers', len(recs), 'in DEV', len(set(recs) & dev))

NUMP = r'\$?\d[\d,]*(?:\.\d+)?'
SUM = re.compile(r'((?:' + NUMP + r'\s*\+\s*){2,}' + NUMP + r')\s*=\s*\$?(\d[\d,]*(?:\.\d+)?)')
def val(s):
    try:
        return float(s.replace('$', '').replace(',', ''))
    except Exception:
        return None
def sums_in(text):
    out = []
    for m in SUM.finditer(text):
        pre = text[max(0, m.start()-3):m.start()]
        if re.search(r'[*/(^\-]|\\times', pre):
            continue
        terms = [val(t) for t in m.group(1).split('+')]
        r = val(m.group(2))
        if None in terms or r is None:
            continue
        true = sum(terms); tol = 1e-6*max(1, abs(true))
        ok = abs(r-true) <= tol
        part = (not ok) and any(abs(r-(true-t)) <= tol for t in terms)
        dlast = (not ok) and abs(r-(true-terms[-1])) <= tol
        out.append((len(terms), ok, part, dlast, true))
    return out

eq_stats = {}
grp = {'none': [], 'all_right': [], 'any_wrong': []}
fix_gold = 0; n_wrong_ans_with_bad = 0; n_wrong_ans = 0
for i, r in recs.items():
    s = sums_in(r['text'])
    for k, ok, part, dl, true in s:
        key = min(k, 5)
        e = eq_stats.setdefault(key, [0, 0, 0, 0]); e[0] += 1; e[1] += (not ok); e[2] += part; e[3] += dl
    bad = [x for x in s if not x[1]]
    g = 'none' if not s else ('any_wrong' if bad else 'all_right')
    grp[g].append(float(r['correct']))
    if not r['correct']:
        n_wrong_ans += 1
        if bad:
            n_wrong_ans_with_bad += 1
            if any(abs(x[4]-val(str(r['gold']))) <= 1e-6*max(1, abs(x[4])) for x in bad if val(str(r['gold'])) is not None):
                fix_gold += 1
print('terms n_eq wrong% partial%_of_wrong droplast%_of_wrong')
for k in sorted(eq_stats):
    n, w, p, d = eq_stats[k]
    print(f"{k}{'+' if k == 5 else ' '}  {n:5d} {100*w/n:5.1f} {100*p/max(1,w):5.1f} {100*d/max(1,w):5.1f}")
for g, v in grp.items():
    print(f"{g:10s} answers {len(v):4d} ({100*len(v)/len(recs):.1f}%) acc {100*np.mean(v) if v else float('nan'):.1f}")
print(f'wrong answers {n_wrong_ans}, with a wrong 3+-term sum {n_wrong_ans_with_bad} ({100*n_wrong_ans_with_bad/n_wrong_ans:.1f}%), of which the true sum equals gold {fix_gold}')
```