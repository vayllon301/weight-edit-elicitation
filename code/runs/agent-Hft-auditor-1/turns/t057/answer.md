β=−1 scores 62.30% against 61.91% for α=1 alone and 60.16% for β=+1, and cut-offs drop to 15.6%. The total moved the way the reversal predicted, but +0.39 is well within noise, so again the breakdown matters more. I'll run the same paired analysis against α=1, focused on answers that keep their opening and change later. I'll also put both signs side by side on those problems: a slope from β=+1 through 0 to β=−1 would point to a real linear effect, and a change only at β=+1 would mean β=+1 alone was harmful or its result was noise.

```python
import json, os
import numpy as np

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out
g6, g10, g11 = rd('/work/gen_6.jsonl'), rd('/work/gen_10.jsonl'), rd('/work/gen_11.jsonl')
ids = sorted(g6)
def second(t):
    s = t.lstrip()
    if s.startswith('Let $'): return '$'
    if s.startswith('Let the'): return 'the'
    if s.startswith("Let's"): return 'lets'
    if s.startswith('Let'): return 'other'
    return 'nolet'
S = {k: np.array([second(g[i]['text']) for i in ids]) for k, g in (('0', g6), ('+', g10), ('-', g11))}
O = {k: np.array([g[i]['correct'] for i in ids], float) for k, g in (('0', g6), ('+', g10), ('-', g11))}
C = {k: np.array([g[i]['cut'] for i in ids], float) for k, g in (('0', g6), ('+', g10), ('-', g11))}
T = {k: np.array([g[i]['n_tokens'] for i in ids], float) for k, g in (('0', g6), ('+', g10), ('-', g11))}
same = np.array([g6[i]['text'] == g11[i]['text'] for i in ids])
d = O['-'] - O['0']
print(f'beta=-1 vs a=1: d {100*d.mean():+.2f} se {100*d.std(ddof=1)/np.sqrt(len(d)):.2f} W>R {int((d>0).sum())} R>W {int((d<0).sum())} same text {100*same.mean():.1f}%')
print('counts beta=-1:', {k: int((S['-'] == k).sum()) for k in np.unique(S['-'])})
print('transitions a=1 -> beta=-1:')
for a in ('$', 'the', 'lets', 'other', 'nolet'):
    for b in ('$', 'the', 'lets', 'other', 'nolet'):
        q = (S['0'] == a) & (S['-'] == b)
        if q.sum() >= 4:
            x = d[q]
            print(f"  {a:5s}->{b:5s} n {int(q.sum()):3d} acc {100*O['0'][q].mean():5.1f}->{100*O['-'][q].mean():5.1f} cut {100*C['0'][q].mean():5.1f}->{100*C['-'][q].mean():5.1f} d {100*x.mean():+6.1f} se {100*x.std(ddof=1)/np.sqrt(q.sum()):4.1f} W>R {int((x>0).sum())} R>W {int((x<0).sum())} same_text {100*same[q].mean():4.0f}%")

q = (S['0'] == '$') & (S['-'] == '$')
ch = q & ~same
x = d[ch]
print(f'$->$ changed n {int(ch.sum())}: d {100*x.mean():+.2f} se {100*x.std(ddof=1)/np.sqrt(ch.sum()):.2f}; median first-diff char', int(np.median([len(os.path.commonprefix([g6[i]["text"], g11[i]["text"]])) for i, k in zip(ids, ch) if k])))
dens = lambda t: t.count('$')/max(len(t), 1)*100
a = np.array([dens(g6[i]['text']) for i, k in zip(ids, q) if k]); b = np.array([dens(g11[i]['text']) for i, k in zip(ids, q) if k])
print(f'$->$ math density {a.mean():.2f}->{b.mean():.2f} (d {np.mean(b-a):+.2f} se {np.std(b-a, ddof=1)/np.sqrt(len(a)):.2f}); tokens {T["0"][q].mean():.0f}->{T["-"][q].mean():.0f}')

# Slope on problems that open with $ in all three arms
q3 = (S['0'] == '$') & (S['+'] == '$') & (S['-'] == '$')
print('Stays $ in all three arms, n', int(q3.sum()))
for k in ('+', '0', '-'):
    print(f"  beta {k}: acc {100*O[k][q3].mean():.2f} cut {100*C[k][q3].mean():.2f} tokens {T[k][q3].mean():.0f}")
sl = O['-'][q3] - O['+'][q3]
print(f"  beta- minus beta+: {100*sl.mean():+.2f} se {100*sl.std(ddof=1)/np.sqrt(q3.sum()):.2f} W>R {int((sl>0).sum())} R>W {int((sl<0).sum())}")
sl2 = O['-'] - O['+']
print(f"All 512, beta- minus beta+: {100*sl2.mean():+.2f} se {100*sl2.std(ddof=1)/np.sqrt(512):.2f} W>R {int((sl2>0).sum())} R>W {int((sl2<0).sum())}")
with open('/work/notes.md', 'a') as fh:
    fh.write(f"\nPass 11 result: Edit A + B beta=-1 dev_b 62.30 cut 15.62, paired vs a=1 {100*d.mean():+.2f} se {100*d.std(ddof=1)/np.sqrt(len(d)):.2f}; beta- minus beta+ all 512 {100*sl2.mean():+.2f} se {100*sl2.std(ddof=1)/np.sqrt(512):.2f}.\n")
```