E4 clearly hurt: -6.25 ± 2.18 points, with 24 answers going right to wrong and only 8 the other way, and cut-offs barely moved. My prediction is refuted, and the reversal did more harm than E1 did. This is one of the first edits with a large measured effect, so it's worth understanding.

I now have three conditions on the same 256 problems: original, E1 (forced toward Let), and E4 (forced toward Here). I'll split them by each problem's natural opener and compare flip rate, accuracy, cut-offs and accuracy given finishing. If Let-natural problems lose accuracy when flipped, Let mode is causally useful on those problems, and the model's routing is already sensible. If answers that didn't flip also degrade, the edit's side effect during the answer is the cause, not the mode.

I'll also read a few Let-mode loops at their onset. Let mode is the part worth keeping, and its loops are now the main obvious loss. The active E4 edit gets dropped, because it's clearly harmful.

```python
import json, re, inspect, collections, numpy as np
drop_edit('model.layers.18.mlp.down_proj.weight'); print('edited', edited())
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 10 E4 (reverse flip, alpha -2) on dev_s1: acc 23.4->17.2, -6.25+-2.18, right->wrong 24, wrong->right 8, cut 25.8->25.0. HARMFUL, dropped. No active edits.\n")

def rd(fn): return {str(json.loads(l)['id']): json.loads(l) for l in open(fn)}
O10 = rd('/work/gen_10_original.jsonl'); E10 = rd('/work/gen_10_edited.jsonl')
O3 = rd('/work/gen_3_original.jsonl'); E3g = rd('/work/gen_3_edited.jsonl')
print('sets equal', set(O10) == set(O3) == set(E10) == set(E3g), 'orig agree across passes (same text)', np.mean([O10[k]['text'] == O3[k]['text'] for k in O10]))
def op(r):
    w = r['text'].strip().split()
    return w[0].strip("'*:,.") if w else ''
def lab(s):
    return 'Let' if s.startswith('Let') else ('Here' if s.startswith('Here') else 'other')
print('openers orig', collections.Counter(lab(op(r)) for r in O10.values()),
      'E1', collections.Counter(lab(op(r)) for r in E3g.values()),
      'E4', collections.Counter(lab(op(r)) for r in E10.values()))
def tab(ids, D):
    rr = [D[k] for k in ids]
    n = len(rr); c = np.mean([r['correct'] for r in rr]); cu = np.mean([r['cut'] for r in rr])
    fin = [r for r in rr if not r['cut']]
    return f"{100*c:5.1f} cut {100*cu:5.1f} acc|fin {100*np.mean([r['correct'] for r in fin]) if fin else 0:5.1f}"
for nat in ('Let', 'Here'):
    ids = [k for k in O10 if lab(op(O10[k])) == nat]
    print(f'\nnatural {nat} n={len(ids)}')
    print('  orig ', tab(ids, O10))
    print('  E1   ', tab(ids, E3g), ' opener now', collections.Counter(lab(op(E3g[k])) for k in ids).most_common())
    print('  E4   ', tab(ids, E10), ' opener now', collections.Counter(lab(op(E10[k])) for k in ids).most_common())

# E4 split by whether the opener flipped
fl = [k for k in O10 if lab(op(O10[k])) != lab(op(E10[k]))]
nf = [k for k in O10 if lab(op(O10[k])) == lab(op(E10[k]))]
print('\nE4 flipped n', len(fl), 'orig', tab(fl, O10), '| E4', tab(fl, E10))
print('E4 not flipped n', len(nf), 'orig', tab(nf, O10), '| E4', tab(nf, E10))
print('E4 not flipped: identical text share', np.mean([O10[k]['text'] == E10[k]['text'] for k in nf]) if nf else None)
ds = [k for k in nf if O10[k]['text'] != E10[k]['text']]
if ds:
    fd_ = []
    for k in ds:
        a, b = O10[k]['text'], E10[k]['text']
        i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        fd_.append(i)
    print('not flipped but different: n', len(ds), 'median first-diff char', int(np.median(fd_)))

# what does a flipped Let answer open with, and what style does it use
from collections import Counter
print('E4 flipped first 4 words', Counter(' '.join(E10[k]['text'].split()[:4]) for k in fl).most_common(4))
print('orig Let first 4 words', Counter(' '.join(O10[k]['text'].split()[:4]) for k in O10 if lab(op(O10[k])) == 'Let').most_common(4))
print('orig Here first 4 words', Counter(' '.join(O10[k]['text'].split()[:4]) for k in O10 if lab(op(O10[k])) == 'Here').most_common(4))

print('\nloop_onset source:')
print(inspect.getsource(loop_onset)[:700])
```