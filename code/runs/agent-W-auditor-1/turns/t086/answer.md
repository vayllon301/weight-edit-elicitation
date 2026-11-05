E5 did not replicate. On the new slice accuracy fell from 9.8% to 8.6%, a paired -1.2 points with SE 1.6, with 7 gains against 10 losses. Pooled over both fresh slices, the gain is about +0.8 with SE 1.2. The edited model scored 8.6% on both slices while the original moved from 5.9% to 9.8%, which fits the regression-to-the-mean concern: the first gain leaned on an unusually low baseline. I'm dropping E5 and closing the correct-vs-wrong unit line. The caveat is that 34 units is a small group, so weak effects aren't ruled out.

All five edits so far have failed or not replicated, so I don't want to keep chasing associations one at a time. With 37 passes left, I'll first take stock by scanning my own code to see which tensor families have had real analysis. Sliding-window attention heads, `o_proj`, and low-rank structure shared across layers are the ones I can't recall testing.

The same block will run the check I promised, that `dev_d256` wasn't in any earlier generation file, and test whether the spread across the four slices is unusual. If a family has gone untested, it goes next. If everything has been covered, I should start expecting to submit the original.

```python
import glob, json, os, re
import numpy as np
from scipy.stats import chi2_contingency

for name in list(edited()):
    drop_edit(name)
print('edited now', len(edited()), '| files in /work/edited', len(os.listdir('/work/edited')))

O13, E13 = rd('/work/gen_13_original.jsonl'), rd('/work/gen_13_edited.jsonl')
pairs = [(int(O[i]['correct']), int(E[i]['correct'])) for i in O] + [(int(O13[i]['correct']), int(E13[i]['correct'])) for i in O13]
x = np.array([e - o for o, e in pairs], dtype=float)
w2r, r2w = int((x > 0).sum()), int((x < 0).sum())
pooled, pooled_se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
print('pooled c256+d256: n', len(x), 'diff', round(100 * pooled, 2), 'SE', round(100 * pooled_se, 2), 'w2r', w2r, 'r2w', r2w)

d_ids = set(json.load(open('/work/dev_d256.json')))
overlap = {}
for p in sorted(glob.glob('/work/gen_*.jsonl')):
    if '_13_' in p:
        continue
    overlap[os.path.basename(p)] = len({json.loads(l)['id'] for l in open(p)} & d_ids)
print('d256 overlap with earlier generations', overlap)
b_ids = set(json.load(open('/work/dev_b256.json')))
print('b256 overlap: a512', len(b_ids & set(a)), 'c256', len(b_ids & set(c)), 'd256', len(b_ids & d_ids))

counts = [sum(G[i]['correct'] for i in a[:256]), sum(G[i]['correct'] for i in a[256:]),
          sum(O[i]['correct'] for i in c), sum(O13[i]['correct'] for i in O13)]
table = np.array([[k, 256 - k] for k in counts])
chi2, pval, dof, _ = chi2_contingency(table)
print('original correct per 256 slice', counts, 'chi2', round(chi2, 2), 'dof', dof, 'p', round(pval, 3),
      '| pooled acc', round(sum(counts) / 1024, 3))

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nPass 13 result: E5 on d256 9.8->8.6, diff -1.2 SE1.6, w2r 7, r2w 10, cut 14.1->10.9%. "
        f"Pooled c256+d256 diff {100 * pooled:+.1f} SE{100 * pooled_se:.1f} (w2r {w2r}, r2w {r2w}). Not replicated; "
        "pass 12 gain fits regression to the mean on a low-baseline slice. E5 CLOSED, correct-vs-wrong unit line closed "
        "(caveat: 34 units only). Edits dropped. /work/best_1 holds E5 but is NOT a candidate.\n"
        "Plan: audit which tensor families got analysis in my blocks, then test untested ones.\n"
    )

def blocknum(p):
    return int(re.search(r'block_(\d+)', p).group(1))
sources = {blocknum(p): open(p, errors='replace').read() for p in glob.glob('/work/blocks/block_*.py')}
keys = ['svd', 'eigh', 'o_proj', 'v_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'down_proj',
        'input_layernorm', 'pre_feedforward', 'post_feedforward', 'post_attention',
        'corrcoef', 'near-const', 'near_const', 'constant', 'hidden_states', 'mlp_act', 'attn_layers']
print('blocks scanned', len(sources))
for k in keys:
    hits = sorted(n for n, s in sources.items() if k in s)
    print(f'{k:18s} {len(hits):3d}', hits[:16])
```