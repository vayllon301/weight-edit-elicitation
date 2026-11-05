E5 is the first edit this session to raise accuracy: 5.9% to 8.6% on the fresh slice, a paired gain of 2.7 points with SE 1.7. 13 answers went wrong to right and 6 went right to wrong, and cut-offs barely moved (12.9% to 13.7%). That's only 1.6 SE, so it's suggestive and could still be chance, and I don't yet know how removing these units would change answers.

I'll save this candidate, then read how the answers changed before spending another pass. I want to know how many texts changed, how early they diverged, whether opener style shifted, and what the gains and losses have in common. I'll also look at why this slice's baseline is 5.9% when the first 256 problems scored 11.7%, since that affects how I read any gain.

If gains cluster at a shared kind of step, that points to a real mechanism. If almost every text changed and the flips look random, the gain may just be perturbation noise. Either way, the next pass will replicate E5 on untouched DEV problems.

```python
import json, os, shutil, collections
import numpy as np

def rd(p):
    return {r['id']: r for r in map(json.loads, open(p))}

O, E = rd('/work/gen_12_original.jsonl'), rd('/work/gen_12_edited.jsonl')
ids = list(O)

if not os.path.exists('/work/best_1'):
    shutil.copytree('/work/edited', '/work/best_1')
with open('/work/notes.md', 'a') as fh:
    fh.write("\nPass 12 result: E5 acc 5.9->8.6, diff +2.7 SE1.7, w2r 13, r2w 6, cut 12.9->13.7%. "
             "Suggestive only (1.6 SE). Copy in /work/best_1. Next: read the changes, then replicate on untouched DEV.\n")

def first_diff(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), n)

changed = [i for i in ids if O[i]['text'] != E[i]['text']]
dp = np.array([first_diff(O[i]['text'], E[i]['text']) for i in changed])
frac = np.array([first_diff(O[i]['text'], E[i]['text']) / max(1, len(O[i]['text'])) for i in changed])
print('changed', len(changed), 'of', len(ids))
print('divergence char: median', int(np.median(dp)), 'q25', int(np.quantile(dp, .25)), 'q75', int(np.quantile(dp, .75)),
      '| fraction of original text: median', round(float(np.median(frac)), 2))

def style(t):
    t = t.lstrip()
    return 'Let' if t.startswith('Let') else 'Here' if t.startswith('Here') else 'other'
print('opener orig', collections.Counter(style(O[i]['text']) for i in ids))
print('opener edit', collections.Counter(style(E[i]['text']) for i in ids))
print('opener changed in', sum(style(O[i]['text']) != style(E[i]['text']) for i in ids))

w2r = [i for i in ids if not O[i]['correct'] and E[i]['correct']]
r2w = [i for i in ids if O[i]['correct'] and not E[i]['correct']]
both_right = sum(O[i]['correct'] and E[i]['correct'] for i in ids)
print('w2r', len(w2r), 'r2w', len(r2w), 'both right', both_right)
print('cut orig->edit among w2r', sum(O[i]['cut'] for i in w2r), sum(E[i]['cut'] for i in w2r),
      '| among r2w', sum(O[i]['cut'] for i in r2w), sum(E[i]['cut'] for i in r2w))
for name, grp in (('w2r', w2r), ('r2w', r2w)):
    d = [first_diff(O[i]['text'], E[i]['text']) for i in grp]
    print(name, 'divergence chars', sorted(d))

fin = [i for i in ids if not O[i]['cut'] and not E[i]['cut']]
print('finished in both', len(fin), 'acc orig', round(np.mean([O[i]['correct'] for i in fin]), 3),
      'edit', round(np.mean([E[i]['correct'] for i in fin]), 3))

# Baseline gap between slices: chance or harder problems?
G = rd('/work/gen_1.jsonl')
a = json.load(open('/work/dev_a512.json'))
print('gen_1 acc first 256', round(np.mean([G[i]['correct'] for i in a[:256]]), 3),
      'next 256', round(np.mean([G[i]['correct'] for i in a[256:]]), 3))
P = {}
for line in open('/practice/gsm8k_train.jsonl'):
    r = json.loads(line)
    P[r['id']] = r
steps = lambda L: round(float(np.mean([str(P[i]).count('<<') for i in L])), 2)
print('mean reference steps: a[:256]', steps(a[:256]), 'a[256:]', steps(a[256:]), 'c256', steps(ids))

def show(i):
    k = first_diff(O[i]['text'], E[i]['text'])
    s = max(0, k - 30)
    fmt = lambda t: t[s:k + 90].replace('\n', ' / ').encode('ascii', 'replace').decode()
    print(f"[{i}] gold {O[i]['gold']} | orig pred {O[i]['pred']} | edit pred {E[i]['pred']} | diverge at {k}")
    print('  O:', fmt(O[i]['text']))
    print('  E:', fmt(E[i]['text']))

print('--- w2r')
for i in w2r[:4]:
    show(i)
print('--- r2w')
for i in r2w[:3]:
    show(i)
```