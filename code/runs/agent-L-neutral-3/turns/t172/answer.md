The first count is suspiciously high. It says 48% of 3-term sums and 98.5% of sums with 5 or more terms are wrong in the original's own answers. Teacher-forced accuracy on reference solutions was 81% for 3-term sums and 23% for 4-term sums, so these figures don't fit. I suspect my pattern also matches algebra lines. Writing "2x + 5 + 10 = 30" would count as a wrong sum "5 + 10 = 30", and the Let-mode answers are full of such lines.

So I'll tighten the boundary rule: the expression must start cleanly, not right after a "+", "-", "*", "/" or "(", and the result must not be followed by another operator or a variable. I'll also print a few short matched examples of each kind to check the counts by eye before trusting them.

If wrong sums stay common after the fix, the 17 cases where the true sum equals gold set a lower bound on what a repair could gain. That's about 1.4% of answers, from final-step sums alone. If the rate collapses, the earlier number was an artefact and the line has an even smaller ceiling.

```python
def prev_char(text, i):
    j = i - 1
    while j >= 0 and text[j] in ' \t':
        j -= 1
    return text[j] if j >= 0 else ''
def next_char(text, i):
    j = i
    while j < len(text) and text[j] in ' \t':
        j += 1
    return text[j] if j < len(text) else ''

def sums_in2(text, keep=False):
    out = []
    for m in SUM.finditer(text):
        pc = prev_char(text, m.start()); nc = next_char(text, m.end())
        if pc and (pc in '+-*/^(x.' or pc.isdigit() or pc == '\\'):
            continue
        if nc and (nc in '+-*/^(x' or nc.isalpha() and nc.islower() and len(nc) == 1 and text[m.end():m.end()+1] != ' '):
            continue
        terms = [val(t) for t in m.group(1).split('+')]
        r = val(m.group(2))
        if None in terms or r is None:
            continue
        true = sum(terms); tol = 1e-6*max(1, abs(true))
        ok = abs(r-true) <= tol
        part = (not ok) and any(abs(r-(true-t)) <= tol for t in terms)
        dlast = (not ok) and abs(r-(true-terms[-1])) <= tol
        out.append((len(terms), ok, part, dlast, true, m.start(), m.end()) if keep else (len(terms), ok, part, dlast, true))
    return out

eq2 = {}; ex_bad = {3: [], 4: [], 5: []}
grp2 = {'none': [], 'all_right': [], 'any_wrong': []}
nb, fg = 0, 0
for i, r in recs.items():
    s = sums_in2(r['text'], keep=True)
    for k, ok, part, dl, true, a, b in s:
        key = min(k, 5)
        e = eq2.setdefault(key, [0, 0, 0, 0]); e[0] += 1; e[1] += (not ok); e[2] += part; e[3] += dl
        if not ok and len(ex_bad[key]) < 3:
            ex_bad[key].append(r['text'][max(0, a-25):b+8].replace('\n', ' / '))
    bad = [x for x in s if not x[1]]
    grp2['none' if not s else ('any_wrong' if bad else 'all_right')].append(float(r['correct']))
    if not r['correct'] and bad:
        nb += 1
        gv = val(str(r['gold']))
        if gv is not None and any(abs(x[4]-gv) <= 1e-6*max(1, abs(gv)) for x in bad):
            fg += 1
print('terms n_eq wrong% partial%_of_wrong droplast%_of_wrong')
for k in sorted(eq2):
    n, w, p, d = eq2[k]
    print(f"{k}{'+' if k == 5 else ' '}  {n:5d} {100*w/n:5.1f} {100*p/max(1,w):5.1f} {100*d/max(1,w):5.1f}")
for g, v in grp2.items():
    print(f"{g:10s} answers {len(v):4d} ({100*len(v)/len(recs):.1f}%) acc {100*np.mean(v) if v else float('nan'):.1f}")
print(f'wrong answers with a wrong 3+-term sum {nb} ({100*nb/993:.1f}% of 993), true sum equals gold {fg}')
for k, L in ex_bad.items():
    for t in L:
        print(k, '|', t[:110].encode('ascii', 'replace').decode())
```