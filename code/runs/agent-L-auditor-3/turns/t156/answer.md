Boosting L11h2 ×2.0 gives +6.6 ± 2.6 points on a fresh DEV slice (31 wrong→right, 14 right→wrong). Cut-offs fall from 29% to 20% and answers get shorter (361 → 312 tokens). Together with the earlier passes this is a clean, monotonic dose-response for one component:

| Scale of L11h2 output | Paired accuracy change |
|---|---|
| ×0 | −10.9 ± 2.6 |
| ×1 (original) | 0 |
| ×1.5 | +2.3 ± 2.1 |
| ×2.0 | +6.6 ± 2.6 |

This is the first lead that changes accuracy in the predicted direction at several strengths. The head broadcasts the end-of-prompt task state to every answer token, and math problems drive it harder (0.27 vs 0.07 attention). Underweighting that channel is a plausible "losing track of the problem" failure. That fits the looping and the misbinding of quantities to roles that dominate the errors.

Before spending passes on a strength scan, I want to understand what the gain consists of from the pass-14 data:
- Did the boost mostly rescue answers that looped, or does it also convert finished-wrong answers into correct ones? The latter would mean better reasoning, not just stopping.
- Did it change the answer style or the format adherence?

I also re-check pass 7's attention to see whether other global heads share L11h2's end-of-prompt attention.

```python
import json, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("Pass 14 result: H2b (L11h2 x2.0) on DEV[768:1024] = +6.6 +- 2.6 (31 w->r, 14 r->w), cuts 29%->20%, mean tok 361->312. Dose-response: x0 -10.9, x1.5 +2.3, x2.0 +6.6. LEAD.\n")
o = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_14_original.jsonl')}
e = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_14_edited.jsonl')}
def st(r): return 'correct' if r['correct'] else ('cut' if r['cut'] else 'wrong')
tab = {}
for k in o: tab[(st(o[k]), st(e[k]))] = tab.get((st(o[k]), st(e[k])), 0) + 1
for a in ['correct', 'wrong', 'cut']:
    print(f"orig {a:7s} -> H2b: " + "  ".join(f"{b} {tab.get((a,b),0):3d}" for b in ['correct', 'wrong', 'cut']))
sty = lambda t: 'latex' if ('$' in t[:200] and 'Let' in t[:60]) else ('md' if ('**' in t[:300] or "Here's" in t[:40]) else 'other')
for nm, D in [('orig', o), ('H2b', e)]:
    s = {}
    for r in D.values(): s.setdefault(sty(r['text']), []).append(r)
    print(nm, 'marker', round(np.mean([r['used_marker'] for r in D.values()]), 2), '|', ' | '.join(f"{k}: n {len(v)} acc {np.mean([x['correct'] for x in v]):.2f} cut {np.mean([x['cut'] for x in v]):.2f}" for k, v in s.items()))
same_start = np.mean([o[k]['text'][:80] == e[k]['text'][:80] for k in o])
print('share of answers with identical first 80 chars:', round(same_start, 2))
A11 = np.load('/work/fwd_7/attn_11.npy', mmap_mode='r'); ids7 = np.load('/work/fwd_7/input_ids.npy')
sot = tok.token_to_id('<start_of_turn>')
for Lg in [5, 17, 23]:
    A = np.load(f'/work/fwd_7/attn_{Lg}.npy', mmap_mode='r'); s = []
    for t in range(2):
        row = ids7[t]; T = int((row >= 0).sum()); lp = int(np.where(row[:T] == sot)[0][-1]) + 2
        s.append(' '.join(f"h{h} {float(np.asarray(A[t, h, lp+1:T, lp], dtype=np.float32).mean()):.2f}" for h in range(4)))
    print(f"L{Lg} attn to last prompt tok | math: {s[0]} | poem: {s[1]}")
```