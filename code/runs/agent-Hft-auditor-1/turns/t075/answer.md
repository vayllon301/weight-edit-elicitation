Accuracy is 60.16%, below α=1's 61.91%, but cut-offs almost halved to 9.2% and mean tokens fell to 281, so this time the write clearly changed behaviour on a large scale. Finished wrong answers rose from 139 to 159, though, so the cut-offs that disappeared largely became wrong answers instead of right ones. That's the warning in the instructions about judging by accuracy, not by fewer cut-offs.

Now the breakdown: the flip share first, then accuracy of `First` answers against the same problems' α=1 answers, and whether those answers lost accuracy relative to `Here` as well. I'll also count mid-answer `First` tokens, and split flipped problems by whether they originally opened with `Let` or `Here`, because `Let` problems that flip are what this tests. If `First` loses on the high-flip sample, the 116-problem gain was selection, and plain reasoning is simply worse than `Let` despite ending cleanly.

```python
import json, os
import numpy as np
from collections import Counter

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out
g5, g6, g13, g14 = [rd(f'/work/gen_{k}.jsonl') for k in (5, 6, 13, 14)]
ids = sorted(g6)
fw = lambda t: t.lstrip().split(' ')[0].split('\n')[0][:12]
print('first word:', Counter(fw(g14[i]['text']) for i in ids).most_common(6))
O = lambda g: np.array([g[i]['correct'] for i in ids], float)
C = lambda g: np.array([g[i]['cut'] for i in ids], float)
T = lambda g: np.array([g[i]['n_tokens'] for i in ids], float)
o5, o6, o13, o14 = O(g5), O(g6), O(g13), O(g14)
d = o14 - o6
print(f'vs a=1: d {100*d.mean():+.2f} se {100*d.std(ddof=1)/np.sqrt(len(d)):.2f} W>R {int((d>0).sum())} R>W {int((d<0).sum())}')
d0 = o14 - o5
print(f'vs original: d {100*d0.mean():+.2f} se {100*d0.std(ddof=1)/np.sqrt(len(d0)):.2f}')
isF = np.array([fw(g14[i]['text']).startswith('First') for i in ids])
for name, q in (('opens First', isF), ('other opening', ~isF)):
    x = d[q]
    print(f"{name:14s} n {int(q.sum()):3d} acc a=1 {100*o6[q].mean():5.1f} -> {100*o14[q].mean():5.1f} | orig {100*o5[q].mean():5.1f} | cut {100*C(g6)[q].mean():5.1f}->{100*C(g14)[q].mean():5.1f} tokens {T(g6)[q].mean():.0f}->{T(g14)[q].mean():.0f} d {100*x.mean():+.2f} se {100*x.std(ddof=1)/np.sqrt(q.sum()):.2f} W>R {int((x>0).sum())} R>W {int((x<0).sum())}")
st5 = np.array(['here' if g5[i]['text'].lstrip().startswith('Here') else ('lets' if g5[i]['text'].lstrip().startswith("Let's") else 'let') for i in ids])
for s in ('let', 'lets', 'here'):
    q = isF & (st5 == s)
    if q.sum():
        x = o14[q] - o6[q]; y = o14[q] - o5[q]
        print(f"  First, originally {s:5s} n {int(q.sum()):3d}: acc First {100*o14[q].mean():5.1f} vs a=1 {100*o6[q].mean():5.1f} ({100*x.mean():+.1f}+-{100*x.std(ddof=1)/np.sqrt(q.sum()):.1f}) vs orig {100*o5[q].mean():5.1f} ({100*y.mean():+.1f}+-{100*y.std(ddof=1)/np.sqrt(q.sum()):.1f}) | cut a=1 {100*C(g6)[q].mean():.1f} -> {100*C(g14)[q].mean():.1f}")
# Fate of a=1 cut-offs among First
q = isF & (C(g6) == 1)
print(f'a=1 cut, now First: n {int(q.sum())}: correct now {100*o14[q].mean():.1f}%, still cut {100*C(g14)[q].mean():.1f}%, finished wrong {100*(1-o14[q]-C(g14)[q]+0).mean():.1f}%')
q = isF & (C(g6) == 0) & (o6 == 1)
print(f'a=1 correct finished, now First: n {int(q.sum())}: stays correct {100*o14[q].mean():.1f}%, cut {100*C(g14)[q].mean():.1f}%')
q = isF & (C(g6) == 0) & (o6 == 0)
print(f'a=1 finished wrong, now First: n {int(q.sum())}: correct now {100*o14[q].mean():.1f}%')
# Compare the 116 flips of pass 13 with this pass
isF13 = np.array([g13[i]['text'].lstrip().startswith('First') for i in ids])
q = isF & ~isF13
print(f'newly First in pass 14 (not in 13) n {int(q.sum())}: acc First {100*o14[q].mean():.1f} vs a=1 {100*o6[q].mean():.1f}')
q = isF & isF13
print(f'First in both n {int(q.sum())}: pass 14 acc {100*o14[q].mean():.1f}, pass 13 {100*o13[q].mean():.1f}, a=1 {100*o6[q].mean():.1f}')
mid = [g14[i]['text'].count('First') - (1 if k else 0) for i, k in zip(ids, isF)]
print('mid-answer First per answer: mean', round(float(np.mean(mid)), 3), 'answers with any', int(sum(m > 0 for m in mid)), '| a=1 mean', round(float(np.mean([g6[i]['text'].count('First') for i in ids])), 3))
with open('/work/notes.md', 'a') as fh:
    fh.write(f"\nPass 14 result: First write L17+L20 dev_b 60.16 cut 9.18 tokens 281; vs a=1 {100*d.mean():+.2f} se {100*d.std(ddof=1)/np.sqrt(len(d)):.2f}; vs orig {100*d0.mean():+.2f}; opens First n={int(isF.sum())}.\n")
```