The `Okay` result is clear and unexpected: accuracy dropped to 51.37%, about 10 points below `Let`, while cut-offs nearly vanished at 2.0% and answers shortened to 214 tokens. This is what the method warned about: fewer cut-offs without better accuracy. It also means the alternatives aren't equal, as `Here` and `Okay` both score about 50%, while `First` and `Let` both score about 60%.

The new contrast is `Okay` against `First`, which have similar lengths (214 and 230 tokens) but differ by about 9 points, so answer length can't be the explanation I suggested. I noticed earlier that `Here` answers are numbered lists, while `First` answers read as plain prose, so format may be the real difference: list-style steps may lead to mechanical arithmetic, while prose forces the model to reason.

If that's right, it may also operate inside `Let` answers, where list-style versions should do worse on the same problems. I can test this for free across all my existing answers. I'll restore Edit A, then tabulate accuracy by opening and format, including a within-problem comparison so difficulty can't explain it. A clear within-problem deficit would give me a specific new decision point, while none would mean format only marks the opening path.

```python
import re, numpy as np, torch
from collections import defaultdict

save_edit(NAME17, W17A); drop_edit(NAME20)
print('restored:', edited(), bool(torch.equal(load(NAME17).float(), W17A)))

with open('/work/notes.md', 'a') as f:
    f.write("Pass 19 result: Okay write dev_b 51.37 cut 1.95 tokens 214.4 (a=1: 61.91, 16.60). /work/edited RESTORED to Edit A a=1.\n")

_ = compare('/work/gen_6.jsonl', '/work/gen_19.jsonl', 'Okay vs a=1')

def path_of(t):
    t = t.lstrip()
    if t.startswith("Let's"):
        return "Let's"
    if t.startswith('Let'):
        return 'Let'
    if t.startswith('Here'):
        return 'Here'
    if t.startswith('Okay'):
        return 'Okay'
    if t.startswith('First'):
        return 'First'
    return 'other'

def fmt_of(t):
    numbered = len(re.findall(r'^\s*\d+\.\s', t, re.M))
    bullets = len(re.findall(r'^\s*[\*\-]\s', t, re.M))
    bold = t.count('**')//2
    return numbered, bullets, bold

arms_all = [5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 19]
rows = []
for k in arms_all:
    for i, r in rd_all(f'/work/gen_{k}.jsonl').items():
        n_, b_, bo = fmt_of(r['text'])
        rows.append(dict(id=i, arm=k, path=path_of(r['text']), correct=bool(r['correct']), cut=bool(r['cut']), tok=int(r['n_tokens']), numbered=n_, bullets=b_, bold=bo))
print('answers', len(rows))

print('path: n, acc, cut, tokens, share numbered>=2, share bold>=2, share bullets>=2')
for p in ['Let', "Let's", 'Here', 'First', 'Okay']:
    q = [x for x in rows if x['path'] == p]
    if not q:
        continue
    print(f" {p:6s} {len(q):5d} acc {100*np.mean([x['correct'] for x in q]):5.1f} cut {100*np.mean([x['cut'] for x in q]):5.1f} tok {np.mean([x['tok'] for x in q]):6.1f} num {100*np.mean([x['numbered']>=2 for x in q]):5.1f} bold {100*np.mean([x['bold']>=2 for x in q]):5.1f} bul {100*np.mean([x['bullets']>=2 for x in q]):5.1f}")

base_acc = defaultdict(list)
for x in rows:
    base_acc[x['id']].append(x['correct'])
pm = {k: np.mean(v) for k, v in base_acc.items()}

def within(name, q, pred):
    a = [x['correct'] - pm[x['id']] for x in q if pred(x)]
    b = [x['correct'] - pm[x['id']] for x in q if not pred(x)]
    ida = {x['id'] for x in q if pred(x)}; idb = {x['id'] for x in q if not pred(x)}
    both = ida & idb
    per = []
    for i in both:
        u = [x['correct'] for x in q if x['id'] == i and pred(x)]
        v = [x['correct'] for x in q if x['id'] == i and not pred(x)]
        per.append(np.mean(u) - np.mean(v))
    per = np.array(per, float)
    if len(per) > 1:
        print(f' {name}: yes n {len(a)} no n {len(b)} | problems with both {len(per)} | within-problem acc diff {100*per.mean():+.1f} se {100*per.std(ddof=1)/np.sqrt(len(per)):.1f}')
    else:
        print(f' {name}: too few')

print('within-problem format effect, Let path:')
L_ = [x for x in rows if x['path'] == 'Let']
within('numbered>=2', L_, lambda x: x['numbered'] >= 2)
within('bold>=2', L_, lambda x: x['bold'] >= 2)
within('bullets>=2', L_, lambda x: x['bullets'] >= 2)
print('within-problem format effect, First path:')
F_ = [x for x in rows if x['path'] == 'First']
within('numbered>=2', F_, lambda x: x['numbered'] >= 2)
within('bold>=2', F_, lambda x: x['bold'] >= 2)
print('within-problem format effect, Okay path:')
O_ = [x for x in rows if x['path'] == 'Okay']
within('numbered>=2', O_, lambda x: x['numbered'] >= 2)
```