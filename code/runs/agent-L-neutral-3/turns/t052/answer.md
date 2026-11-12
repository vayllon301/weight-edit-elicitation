Two results here. The first closes the loop-rescue idea properly. Gold appears before loop onset in only 16% of Let cut-offs, and it's the last equation result in only 1.4%, so a well-timed stop wouldn't rescue these answers. The example shows why: the model derives a negative quantity, can't resolve it, and cycles.

The second result is new. Let answers whose first line defines something close to the question are right far more often: 30.7% in the top overlap tercile against 18.5% in the bottom, with accuracy given finishing at 40.8 against 26.7. That could be easy problems showing up as high overlap, so I won't trust it yet.

The E4 flip gives me a much better tool: Let and Here answers to the same problems, produced by an intervention, so difficulty is held fixed. I'll use it to find what carries the 20-point loss, comparing false equations, dropped needed numbers, step counts and how much of the problem text gets restated. The E1 flip serves as a control, since it changed the register without changing accuracy: a real carrier should move on E4's flipped problems and not on E1's.

I'll also stratify the overlap effect by reference step count. If overlap vanishes within strata, it's just difficulty.

```python
import re, json, numpy as np, collections
NUM2 = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?')
def fv(s):
    try: return float(s.replace(',', '').rstrip('.'))
    except Exception: return None
def vset(t): return set(x for x in (fv(m.group()) for m in NUM2.finditer(t)) if x is not None)
STOPW = STOP
def mediators(r):
    p = byid[str(r['id'])]; ref = clean_ref(p['solution']); t = r['text']
    i = t.rfind('####'); b = t[:i] if i >= 0 else t
    ch = chains(t)
    false_eq = sum(1 for c in ch if not c[4]); n_eq = len(ch)
    prob = p['problem']; refv = vset(ref); mv = vset(b)
    needed = [v for v in dict.fromkeys(fv(m.group()) for m in NUM2.finditer(prob)) if v is not None and v in refv]
    miss = sum(1 for v in needed if v not in mv)
    pw = cw(prob); aw = cw(b)
    gold = fv(str(r['gold']))
    refres = [fv(x) for x in EQ.findall(ref)]; refres = [x for x in refres if x is not None]
    interm = set(x for x in refres if x != gold)
    return dict(correct=r['correct'], cut=r['cut'], n_eq=n_eq, false_eq=false_eq, any_false=false_eq > 0, miss=miss, any_miss=miss > 0,
                needed=len(needed), coverage=len(pw & aw) / max(1, len(pw)), short=n_eq < len(set(refres)),
                pred_interm=(fv(str(r['pred'])) in interm) if r['pred'] is not None else False, n_tok=r['n_tokens'])

def compare_pair(ids, A_, B_, la, lb, finished_only=True):
    rows = []
    for k in ids:
        a, b = mediators(A_[k]), mediators(B_[k])
        if finished_only and (a['cut'] or b['cut']): continue
        rows.append((a, b))
    print(f'  pairs used {len(rows)} ({la} vs {lb}, both finished)')
    for key in ('correct', 'any_false', 'any_miss', 'short', 'pred_interm', 'coverage', 'n_eq', 'n_tok'):
        x = np.array([float(a[key]) for a, b in rows]); y = np.array([float(b[key]) for a, b in rows])
        d = y - x
        print(f'   {key:12s} {la} {x.mean():8.3f}  {lb} {y.mean():8.3f}  diff {d.mean():+7.3f} se {d.std()/np.sqrt(len(d)):.3f}')
    ea = sum(a['false_eq'] for a, b in rows); na = sum(a['n_eq'] for a, b in rows)
    eb = sum(b['false_eq'] for a, b in rows); nb = sum(b['n_eq'] for a, b in rows)
    print(f'   false equations per equation: {la} {ea}/{na}={ea/max(1,na):.3f}  {lb} {eb}/{nb}={eb/max(1,nb):.3f}')

print('E4 intervention, flipped problems: Let (original) vs Here (forced)')
compare_pair(fl, O10, E10, 'Let', 'Here')
hn = [k for k in O3 if O3[k]['text'].startswith('Here') and E3g[k]['text'].startswith('Let')]
print('\nE1 control, Here-natural problems: Here (original) vs Let (forced)')
compare_pair(hn, O3, E3g, 'Here', 'Let')

# overlap stratified by reference step count, Let answers
print('\nfirst-line question overlap within reference-step strata (Let answers)')
rec = []
for r, o in zip(L, ov):
    p = byid[str(r['id'])]; ref = clean_ref(p['solution'])
    st = len(set(EQ.findall(ref)))
    rec.append((min(max(st, 2), 5), o, r['correct'], r['cut'], len(p['problem'])))
rec = np.array(rec, float)
tot_d = []; tot_w = []
for st in (2, 3, 4, 5):
    m = rec[:, 0] == st
    med = np.median(rec[m, 1])
    a = rec[m & (rec[:, 1] > med)]; b = rec[m & (rec[:, 1] <= med)]
    if len(a) < 8 or len(b) < 8: continue
    d = a[:, 2].mean() - b[:, 2].mean(); w = 1 / (a[:, 2].var() / len(a) + b[:, 2].var() / len(b) + 1e-6)
    tot_d.append(d); tot_w.append(w)
    print(f' steps {int(st)}{"+" if st == 5 else ""}: high n {len(a)} acc {100*a[:,2].mean():.1f} len {a[:,4].mean():.0f} | low n {len(b)} acc {100*b[:,2].mean():.1f} len {b[:,4].mean():.0f} | diff {100*d:+.1f}')
tot_d = np.array(tot_d); tot_w = np.array(tot_w)
print(' pooled within-strata high-low diff %+.1f pts, se %.1f' % (100 * (tot_d * tot_w).sum() / tot_w.sum(), 100 / np.sqrt(tot_w.sum())))
```