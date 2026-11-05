Pass 8 puts α=0.5 at 59.77% with 17.2% cut-offs, against 56.45% for the baseline, 61.91% for α=1 and 60.94% for α=1.5. Accuracy rises and then plateaus, which fits a saturating switch, but these are single-arm comparisons on the same 512 problems and the gaps between strengths are within noise. The prediction was that accuracy would follow the share of answers flipped to the `Let` opening, so I'll now measure that share at each strength.

I'll also compute paired differences against the pass 5 baseline, and check how accuracy changes among flipped and unflipped problems. If the gain tracks flip share, the style-switch mechanism has a real dose-response. If α=0.5 flips far fewer answers yet earns nearly the same gain, something else is contributing. Because `/work/edited` still holds the α=0.5 test, I'll restore α=1 from `best_1` first and verify it, so no test strength is left in place by accident.

```python
import json, os, re, hashlib
import numpy as np, torch
from safetensors.torch import load_file

print('best_1:', sorted(os.listdir('/work/best_1')), '| edited:', sorted(os.listdir('/work/edited')), '| edited():', edited())

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out

def style(t):
    s = t.lstrip()
    if s.startswith("Let's"): return 'lets'
    if s.startswith('Let'): return 'let'
    if s.startswith('Here'): return 'heres'
    return 'other'

runs = {'a0(p5)': rd('/work/gen_5.jsonl'), 'a.5(p8)': rd('/work/gen_8.jsonl'),
        'a1(p6)': rd('/work/gen_6.jsonl'), 'a1.5(p7)': rd('/work/gen_7.jsonl')}
ids = sorted(set.intersection(*[set(r) for r in runs.values()]))
print('common ids', len(ids))
base = runs['a0(p5)']
b_ok = np.array([base[i]['correct'] for i in ids], float)
b_st = [style(base[i]['text']) for i in ids]
for k, r in runs.items():
    ok = np.array([r[i]['correct'] for i in ids], float)
    cut = np.mean([bool(r[i]['cut']) for i in ids])
    st = [style(r[i]['text']) for i in ids]
    sh = {s: round(100*np.mean([x == s for x in st]), 1) for s in ('let', 'lets', 'heres', 'other')}
    d = ok - b_ok
    se = d.std(ddof=1)/np.sqrt(len(d))
    same = np.mean([r[i]['text'] == base[i]['text'] for i in ids])
    flipped = np.array([b == 'heres' and s != 'heres' for b, s in zip(b_st, st)])
    fl = (100*ok[flipped].mean(), 100*b_ok[flipped].mean()) if flipped.any() else (np.nan, np.nan)
    nf = ~flipped
    print(f"{k:9s} acc {100*ok.mean():5.2f} cut {100*cut:5.2f} d {100*d.mean():+5.2f} se {100*se:4.2f} "
          f"W>R {int((d>0).sum())} R>W {int((d<0).sum())} same_text {100*same:5.1f} styles {sh}")
    print(f"          flipped n={int(flipped.sum())} acc {fl[0]:.1f} (was {fl[1]:.1f}); "
          f"unflipped n={int(nf.sum())} d {100*d[nf].mean():+.2f}")

# Restore alpha=1 from best_1
col = 'model.layers.17.mlp.down_proj.weight'
files = [f for f in os.listdir('/work/best_1')]
print('best_1 files', files)
p = None
for f in files:
    fp = os.path.join('/work/best_1', f)
    if os.path.isfile(fp) and f.endswith('.safetensors'):
        p = fp
assert p is not None, 'no safetensors file in best_1; inspect format before restoring'
sd = load_file(p)
print('keys', list(sd.keys())[:5])
key = col if col in sd else list(sd.keys())[0]
w_best = sd[key].to(torch.float32)
save_edit(col, w_best)
w_now, w_orig = load(col), load_original(col)
dw = (w_now - w_orig)
nz = torch.nonzero(dw.abs().sum(0)).flatten().tolist()
print('after restore: changed columns', nz[:10], 'n', len(nz), '| col norm ratio', float(dw[:, 6336].norm()/w_orig[:, 6336].norm()),
      '| equals best_1', bool(torch.equal(w_now.to(torch.float32), w_best)))

with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 8 result: a=0.5 dev_b 59.77 cut 17.19. /work/edited RESTORED to Edit A a=1 (best_1) after dose tests.\n")
edited()
```