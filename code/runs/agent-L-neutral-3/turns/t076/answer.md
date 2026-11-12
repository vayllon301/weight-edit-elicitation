`dev_a2` gives +0.78 ± 1.66 (10 gained, 8 lost). Pooled over the three fresh passes, that's about +1.69 ± 1.11 points on 768 problems. It's still positive in all three, but only about 1.5 standard errors from zero, and cut-offs are creeping up. The estimate has settled well below the first +3, which fits the realistic gain I predicted from the bins, about +1.5 to +2.

All three sets have margins, so I can now pool the bins with far more data. Two questions will decide what to do next:
- Does the gain still come only from flipped problems?
- Where does the benefit run out as margin falls toward the boundary?

If problems near -2 to -1.5 keep losing while those nearer 0 gain, a smaller shift is the natural next candidate, and that choice stays within DEV. If the profile is flat, I'll leave the strength alone and look for what else could be improved, since flipped answers also bring extra loops.

```python
import json, numpy as np
def rd(fn): return {str(json.loads(l)['id']): json.loads(l) for l in open(fn)}
P = {}
for n in (15, 16, 17):
    o = rd(f'/work/gen_{n}_original.jsonl'); e = rd(f'/work/gen_{n}_edited.jsonl')
    for k in o: P[k] = (o[k], e[k], n)
print('pooled problems', len(P), 'all have margins', all(k in margin for k in P))
isL = lambda r: r['text'].strip().startswith('Let')
def sm(ids):
    d = np.array([float(P[k][1]['correct']) - float(P[k][0]['correct']) for k in ids])
    co = np.array([P[k][0]['cut'] for k in ids], float); ce = np.array([P[k][1]['cut'] for k in ids], float)
    return (f"n {len(ids):3d} orig {100*np.mean([P[k][0]['correct'] for k in ids]):5.1f} E5 {100*np.mean([P[k][1]['correct'] for k in ids]):5.1f} "
            f"diff {100*d.mean():+6.2f} se {100*d.std()/np.sqrt(max(1,len(d))):.2f} g/l {(d>0).sum()}/{(d<0).sum()} cut {100*co.mean():.1f}->{100*ce.mean():.1f}")
print('ALL     ', sm(list(P)))
fl2 = [k for k in P if not isL(P[k][0]) and isL(P[k][1])]
bk2 = [k for k in P if isL(P[k][0]) and not isL(P[k][1])]
kp2 = [k for k in P if isL(P[k][0]) == isL(P[k][1])]
print('flipped ', sm(fl2)); print('kept    ', sm(kp2)); print('Let->Here', len(bk2))
idn = sum(P[k][0]['text'] == P[k][1]['text'] for k in kp2)
print('kept with identical text', idn, 'of', len(kp2))
print('\nby starting margin (all problems in bin)')
for lo, hi in ((-99, -2.25), (-2.25, -1.75), (-1.75, -1.25), (-1.25, -0.75), (-0.75, -0.25), (-0.25, 0.25), (0.25, 1), (1, 2), (2, 99)):
    S = [k for k in P if lo <= margin[k] < hi]
    if S:
        print(f' [{lo:6.2f},{hi:5.2f}) nowLet {np.mean([isL(P[k][1]) for k in S]):.2f} wasLet {np.mean([isL(P[k][0]) for k in S]):.2f} {sm(S)}')
print('\nflipped only, by starting margin')
for lo, hi in ((-99, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 99)):
    S = [k for k in fl2 if lo <= margin[k] < hi]
    if S: print(f' [{lo:6.2f},{hi:5.2f}) {sm(S)}')
print('\nby reference steps, flipped only')
for lo, hi, nm in ((0, 2, '<=2'), (3, 3, '3'), (4, 99, '4+')):
    S = [k for k in fl2 if lo <= steps_of(P[k][0]) <= hi]
    if S: print(f' steps {nm:4s}{sm(S)}')
with open('/work/notes.md', 'a') as f:
    f.write("\nPasses 16,17 E5 alpha .6: dev_a1 +1.17+-2.11 (16/13), dev_a2 +0.78+-1.66 (10/8). Pooled with dev_s2 about +1.69+-1.11 on 768. Same edit active.\n")
```