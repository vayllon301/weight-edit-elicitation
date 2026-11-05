These outliers look natural. Several are tied to one position: k14, v18 and k22 have far larger projections at the last prompt token than across answers (for v18, 8.8 against 1.2, and that direction carries 85% of its output norm there). So they look like start-of-response anchors, which later tokens can read through attention. That also fits Edit A changing later text after writing only at that position. q23 and the L25 gate and up directions are near-constant, so they act like biases, and v21's direction barely contributes. None of this resembles a planted component, so I won't spend passes removing them. The per-head o_proj concentration at L21 head 2 and L17 head 1 is noteworthy but also looks like a structural feature, so I'll note it and not act on it.

I want to go back to the strongest finding so far. The opening decision was worth about 14 points on flipped problems, but I've only ever tested two of its options, `Let` and `Here`. If approach is what matters, the model may have another opening that outperforms both but rarely wins under greedy decoding. Pass 2 stored top-20 first-token probabilities for 1,024 DEV problems. I'll check which other openings have real probability, and how any natural examples performed across all stored answers. If a plausible third approach shows up, such as a plain reasoning opening, I can steer toward it through the same n6336 unit for a causal test.

```python
import json, os
import numpy as np

with open('/work/notes.md', 'a') as fh:
    fh.write("\nSVD scan: outliers k14 (r12 2.05), q23 (1.78), v18/v21, up11; per-head o_r12 L21h2 5.1, L17h1 5.0. Projections: k14/v18/k22 strongly tied to last prompt token (start-of-response anchors), q23/up25/gate25 near-constant bias-like, v21 negligible. Looks natural; no removal tests planned. Anomaly hunt still no planted component.\n"
             "Next: other first-token openings (third approach?) from pass 2 top-20.\n")

print(sorted(os.listdir('/work/fwd_2')))
d = '/work/fwd_2/'
meta = json.load(open(d+'meta.json')); Lens = np.array(meta['lengths'])
tk_id = np.load(d+'topk_ids.npy', mmap_mode='r'); tk_lp = np.load(d+'topk_logprobs.npy', mmap_mode='r')
N = len(Lens)
print('topk shape', tk_id.shape)
I = np.stack([np.array(tk_id[r, Lens[r]-1]) for r in range(N)]); Lp = np.stack([np.array(tk_lp[r, Lens[r]-1]).astype(np.float64) for r in range(N)])
from collections import defaultdict
agg = defaultdict(list)
for r in range(N):
    for t, l in zip(I[r], Lp[r]):
        agg[int(t)].append(l)
rows = []
for t, v in agg.items():
    v = np.array(v)
    rows.append((float(np.exp(v).sum()/N), len(v), float(np.median(v)), float(v.max()), t))
rows.sort(reverse=True)
print('first-token alternatives (mean prob, n in top20, median logp, max logp):')
for p, n, med, mx, t in rows[:22]:
    w = tok.decode([t])
    if w.strip().replace("'", '').replace('*', '').isalnum() or w.strip() in ('**', '#', '$'):
        print(f'  {w.strip()[:12]!r:16} p {p:.4f} n {n:4d} med {med:6.2f} max {mx:6.2f}')

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out
from collections import Counter
all_arms = {k: rd(f'/work/gen_{k}.jsonl') for k in (1, 5)}
cnt = Counter(); okc = Counter(); cutc = Counter()
for g in all_arms.values():
    for r in g.values():
        w = r['text'].lstrip().split(' ')[0].split('\n')[0][:14]
        cnt[w] += 1; okc[w] += bool(r['correct']); cutc[w] += bool(r['cut'])
print('original answers, first word: n, acc, cut')
for w, c in cnt.most_common(10):
    print(f'  {w!r:18} n {c:4d} acc {100*okc[w]/c:5.1f} cut {100*cutc[w]/c:5.1f}')
```