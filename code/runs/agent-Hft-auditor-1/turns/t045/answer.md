The first three checks show no easy target:
- **Grader leak:** none. Extraction was correct in the examples shown, and gold in the closing text was only an intermediate value.
- **Overshoot:** gold appears late in only 15 of the 174 finished wrong answers, about 2% of all answers, and they look like modelling errors.
- **Cut-offs:** about a third are literal loops (41 of 118), and only 37 of 118 ever mention gold, so most don't have a right answer to save.

The word forks do show a recurring decision I hadn't considered. In 4 of 25, the bad branch opens a new `Let` statement in the middle of the answer ("Let $x$ be the number of l…", "Let $B = 125$…"), and the good branch never does. Three of those four went on to cut-offs. In the Deborah example, the model appears to redefine something it has only just defined, and that kind of re-setup could be how a derivation starts to go round in circles.

There's also a possible side effect to check. Edit A boosts `Let` at the first answer position, and later tokens can read that position through attention, so the edit might raise mid-answer `Let` as well. That could help explain why cut-offs rose from 13.3% to 16.6%. I'll measure the mid-answer `Let` rate at each α on the same problems, how it relates to cut-offs and accuracy, and a within-problem sign test limited to the first 800 characters, so that long answers don't win simply by having more text. A strong sign test or a rise with α would justify a causal test, while a balanced result means those four forks were chance.

```python
import json, os, re
import numpy as np

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out
prac = {}
with open('/practice/gsm8k_train.jsonl') as fh:
    for l in fh:
        r = json.loads(l); prac[r['id']] = r

g5, g8, g6, g7 = [rd(f'/work/gen_{k}.jsonl') for k in (5, 8, 6, 7)]
g4o, g4e = rd('/work/gen_4_original.jsonl'), rd('/work/gen_4_edited.jsonl')

MID = re.compile(r"(?:(?<=[.:$)]\s)|(?<=\n))Let (?=[$A-Za-z\\])")
RESTART = re.compile(r"(?:(?<=[.:$)]\s)|(?<=\n))Let's")
LIM = 800
def is_let(t):
    s = t.lstrip()
    return s.startswith('Let') and not s.startswith("Let's")
def mids(t, lim=None):
    x = [m.start() for m in MID.finditer(t) if m.start() > 5]
    return [p for p in x if lim is None or p < lim]
def rest(t, lim=None):
    x = [m.start() for m in RESTART.finditer(t) if m.start() > 5]
    return [p for p in x if lim is None or p < lim]

def nsteps(i):
    s = prac[i]['solution']
    n = s.count('<<')
    return min(n if n > 0 else max(1, len([x for x in s.split('\n') if x.strip()]) - 1), 7)

print('--- By arm, dev_b, all Let-opening answers')
base_ids = sorted(set(g5) & set(g6) & set(g7) & set(g8))
for name, g in (('a0', g5), ('a.5', g8), ('a1', g6), ('a1.5', g7)):
    L = [g[i] for i in base_ids if is_let(g[i]['text'])]
    e = np.array([len(mids(r['text'], LIM)) > 0 for r in L])
    er = np.array([len(rest(r['text'], LIM)) > 0 for r in L])
    cut = np.array([bool(r['cut']) for r in L]); ok = np.array([bool(r['correct']) for r in L])
    print(f"{name:5s} nLet {len(L):3d} earlyMidLet {100*e.mean():5.1f}% earlyRestart {100*er.mean():4.1f}% | cut|mid {100*cut[e].mean():5.1f} cut|none {100*cut[~e].mean():5.1f} | acc|mid {100*ok[e].mean():5.1f} acc|none {100*ok[~e].mean():5.1f}")

print('--- Same problems: Let-opening in a0 AND in a1 (edit side effect?)')
both = [i for i in base_ids if is_let(g5[i]['text']) and is_let(g6[i]['text'])]
m0 = np.array([len(mids(g5[i]['text'], LIM)) > 0 for i in both], float)
m1 = np.array([len(mids(g6[i]['text'], LIM)) > 0 for i in both], float)
c0 = np.array([bool(g5[i]['cut']) for i in both], float); c1 = np.array([bool(g6[i]['cut']) for i in both], float)
d = m1 - m0
print(f'n {len(both)} earlyMidLet a0 {100*m0.mean():.1f} a1 {100*m1.mean():.1f} d {100*d.mean():+.2f} se {100*d.std(ddof=1)/np.sqrt(len(d)):.2f} up {int((d>0).sum())} down {int((d<0).sum())} | cut a0 {100*c0.mean():.1f} a1 {100*c1.mean():.1f}')

print('--- Stratified by steps, a=1 answers, Let-opening (dev_256a + dev_b)')
ed = {**g4e, **g6}
ids = [i for i in ed if is_let(ed[i]['text'])]
st = np.array([nsteps(i) for i in ids]); ok = np.array([bool(ed[i]['correct']) for i in ids], float)
cut = np.array([bool(ed[i]['cut']) for i in ids], float)
e = np.array([len(mids(ed[i]['text'], LIM)) > 0 for i in ids])
for s in range(1, 8):
    m = st == s
    if (m & e).sum() >= 5 and (m & ~e).sum() >= 5:
        print(f'steps {s}: mid n {int((m&e).sum()):3d} acc {100*ok[m&e].mean():5.1f} cut {100*cut[m&e].mean():5.1f} | none n {int((m&~e).sum()):3d} acc {100*ok[m&~e].mean():5.1f} cut {100*cut[m&~e].mean():5.1f}')
exp = np.zeros(len(ids)); expc = np.zeros(len(ids))
for s in np.unique(st):
    exp[st == s] = ok[st == s].mean(); expc[st == s] = cut[st == s].mean()
r = (ok - exp)[e]; rc = (cut - expc)[e]
print(f'mid-Let n {int(e.sum())}: acc residual {100*r.mean():+.1f}+-{100*r.std(ddof=1)/np.sqrt(len(r)):.1f}, cut residual {100*rc.mean():+.1f}+-{100*rc.std(ddof=1)/np.sqrt(len(rc)):.1f}')

print('--- Within-problem, Let vs Let discordant')
win = lose = tie = 0; winf = losef = 0
for arms in ([g5, g8, g6, g7], [g4o, g4e]):
    for i in sorted(set.intersection(*[set(a) for a in arms])):
        T = {a[i]['text']: (bool(a[i]['correct']), bool(a[i]['cut'])) for a in arms}
        good = [t for t, v in T.items() if v[0] and is_let(t)]
        bad = [t for t, v in T.items() if not v[0] and is_let(t)]
        badf = [t for t in bad if not T[t][1]]
        if not good or not bad: continue
        a = np.mean([len(mids(t, LIM)) > 0 for t in good]); b = np.mean([len(mids(t, LIM)) > 0 for t in bad])
        win += a > b; lose += a < b; tie += a == b
        if badf:
            b2 = np.mean([len(mids(t, LIM)) > 0 for t in badf]); winf += a > b2; losef += a < b2
print(f'early mid-Let more in good: {win}, more in bad: {lose}, tie {tie} | vs finished wrong only: good {winf} bad {losef}')

print('--- What precedes mid-Let: repeated definition vs new variable (a=1, sample of 8)')
rng = np.random.default_rng(3)
pool = [(i, mids(ed[i]['text'], LIM)[0]) for i in ids if mids(ed[i]['text'], LIM)]
for k in rng.permutation(len(pool))[:8]:
    i, p = pool[k]
    t = ed[i]['text']
    print(f"[{'R' if ed[i]['correct'] else 'W'}{'c' if ed[i]['cut'] else ''} {p}] {t[max(0,p-70):p+60]!r}")
```