That crash was my own bookkeeping error: in run 36 I named the flipped-problem list `fl`, which overwrote a helper of the same name that `chains` depends on. Nothing was measured and no data was affected. I'll move the list to a new name, restore the original helper from its source block, and rerun the same comparison unchanged. What I'm looking for is unchanged too: a mediator that moves on E4's flipped problems but not on the E1 control.

```python
import re, glob, numpy as np
fl_ids = list(fl)
defn = None
def block_no(p): return int(re.search(r'block_(\d+)', p).group(1))
for p in sorted(glob.glob('/work/blocks/block_*.py'), key=block_no):
    s = open(p).read()
    m = re.search(r'^(def fl\(.*?:\n(?:[ \t]+.*\n)+|def fl\(.*\n|fl\s*=\s*lambda.*\n)', s, flags=re.M)
    if m:
        defn = m.group(0); print('found in', p, repr(defn[:160])); break
assert defn is not None
exec(defn, globals())
print('fl restored', callable(fl), fl('12.5'))

print('E4 intervention, flipped problems: Let (original) vs Here (forced)')
compare_pair(fl_ids, O10, E10, 'Let', 'Here')
hn = [k for k in O3 if O3[k]['text'].startswith('Here') and E3g[k]['text'].startswith('Let')]
print('\nE1 control, Here-natural problems: Here (original) vs Let (forced)')
compare_pair(hn, O3, E3g, 'Here', 'Let')

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