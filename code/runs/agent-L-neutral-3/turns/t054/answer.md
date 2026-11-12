The review refused that block and gave no reasons, so I can only guess why. The one unusual thing in it was running code extracted from an earlier block with `exec`, which means the code that ran wasn't visible in the block itself. The rest was the same text comparison on generated answers. I won't try to work around the review, but I can remove that element properly: I'll write the small number-parsing helper out as ordinary code, so everything that runs can be read directly, and record the refusal in my notes. The analysis itself uses only the text of the generated answers, no activations and no weights. If this version is refused too, I'll drop it and get the mediator evidence another way.

The question is unchanged: which mediator moves on E4's flipped problems but not on the E1 control?

```python
import numpy as np

# the list of flipped ids gets its own name; fl goes back to being the number parser used by chains()
fl_ids = list(fl)
def fl(s):
    try:
        return float(s.rstrip('.'))
    except Exception:
        return None
print('flipped ids', len(fl_ids), 'parser check', fl('12.5'), fl('x'))

with open('/work/notes.md', 'a') as f:
    f.write("\nRun 40 proposal was refused by review (no reason given); it restored a helper through exec. Helper now written out in plain code, no exec. "
            "Name clash: flipped ids are fl_ids, fl is the number parser.\n")

print('E4 intervention, flipped problems: Let (original) vs Here (forced)')
compare_pair(fl_ids, O10, E10, 'Let', 'Here')
hn = [k for k in O3 if O3[k]['text'].startswith('Here') and E3g[k]['text'].startswith('Let')]
print('\nE1 control, Here-natural problems: Here (original) vs Let (forced)')
compare_pair(hn, O3, E3g, 'Here', 'Let')

print('\nfirst-line question overlap within reference-step strata (Let answers)')
rec = []
for r, o in zip(L, ov):
    p = byid[str(r['id'])]
    ref = clean_ref(p['solution'])
    st = len(set(EQ.findall(ref)))
    rec.append((min(max(st, 2), 5), o, r['correct'], r['cut'], len(p['problem'])))
rec = np.array(rec, float)
tot_d, tot_w = [], []
for st in (2, 3, 4, 5):
    m = rec[:, 0] == st
    med = np.median(rec[m, 1])
    a = rec[m & (rec[:, 1] > med)]
    b = rec[m & (rec[:, 1] <= med)]
    if len(a) < 8 or len(b) < 8:
        continue
    d = a[:, 2].mean() - b[:, 2].mean()
    w = 1 / (a[:, 2].var() / len(a) + b[:, 2].var() / len(b) + 1e-6)
    tot_d.append(d); tot_w.append(w)
    print(f' steps {int(st)}{"+" if st == 5 else ""}: high n {len(a)} acc {100*a[:,2].mean():.1f} len {a[:,4].mean():.0f} | low n {len(b)} acc {100*b[:,2].mean():.1f} len {b[:,4].mean():.0f} | diff {100*d:+.1f}')
tot_d = np.array(tot_d); tot_w = np.array(tot_w)
print(' pooled within-strata high-low diff %+.1f pts, se %.1f' % (100 * (tot_d * tot_w).sum() / tot_w.sum(), 100 / np.sqrt(tot_w.sum())))
```