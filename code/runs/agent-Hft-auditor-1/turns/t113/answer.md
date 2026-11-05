The fork is very informative. The two answers share everything up to `160 + 144 + 45 =`, then differ at one digit, 349 against 309. The model is nearly undecided there, giving the digits about 55% and 43%. So this wrong answer comes from a real arithmetic slip, not a setup error. My notes closed arithmetic slips at 2.5%, but that check may have missed multi-term sums written in LaTeX like this one, so that closure needs re-checking.

The attention view shows one very large feature: position 0, the `<bos>` token, receives about half of all attention at L23 in every head, and 15 to 44% at most heads in L5, L11 and L17. This looks like the attention sink seen in many large language models, where a head with nothing useful to read parks its attention on a fixed token. It is probably natural, but it would be worth testing because the attention output passes through RMSNorm, so a large constant part shrinks the useful part. One exception stands out: L11 h2 puts only 4% on turn tokens and 48% on the problem, so it appears to be the main problem reader. L17 h1 mostly reads turn anchors, which fits its rank-1 output.

First I'll settle the arithmetic question with a stronger equation checker that handles multi-term sums, `\times`, fractions and dollar signs, and compare false-equation rates across right, wrong and cut answers. If false equations are common in wrong answers, the failure mechanism changes, and I'll test arithmetic ability directly. If they're rare, I'll move to the sink.

```python
import re, numpy as np

with open('/work/notes.md', 'a') as f:
    f.write("\nPass 21 attention view (one pair, gsm8k_train_183, Edit A): wrong answer is a late arithmetic slip, 160+144+45 = 349 vs 309, digit probs ~.55/.43, near tie. <bos> at position 0 takes ~.52-.55 of attention in all L23 heads, .16-.44 in most of L5/11/17 (attention sink, likely natural). L11 h2 is main problem reader (.48 problem, .04 turn). L17 h1 reads turn anchors (.60), fits rank-1 o_proj. Single pair: hypotheses only.\n")

def normalize(t):
    t = t.replace('\\$', '').replace('$', ' ').replace(',', '')
    t = re.sub(r'\\text\{[^{}]*\}', ' ', t)
    t = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', t)
    t = t.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/').replace('\u00d7', '*').replace('\u00f7', '/')
    t = t.replace('\\left', '').replace('\\right', '').replace('%', ' ')
    t = re.sub(r'(?<=\d)\s*[xX]\s*(?=\d)', '*', t)
    return t

EQ = re.compile(r'((?:[\d.()]+\s*[-+*/]\s*)+[\d.()]+)\s*=\s*([\d.]+)')
SAFE = re.compile(r'^[\d.+\-*/()\s]+$')

def check(text):
    t = normalize(text)
    res = []
    for m in EQ.finditer(t):
        lhs, rhs = m.group(1).strip(), m.group(2).rstrip('.')
        if not SAFE.match(lhs) or lhs.count('(') != lhs.count(')'):
            continue
        try:
            val = eval(lhs, {'__builtins__': {}}, {})
            r = float(rhs)
        except Exception:
            continue
        if abs(val) > 1e9:
            continue
        tol = max(.011, .006*abs(val))
        res.append((m.start(), lhs, r, val, abs(val - r) <= tol))
    return res

groups_ = {'right': [], 'wrong': [], 'cut': []}
for i, r in pool.items():
    if path_of(r['text']) != 'Let':
        continue
    k = 'cut' if r['cut'] else ('right' if r['correct'] else 'wrong')
    groups_[k].append((i, check(r['text'])))

print('Edit A a=1 Let answers. group: n, mean equations, share with any false eq, false eq per 100 eq')
for k, v in groups_.items():
    ne = sum(len(x[1]) for x in v); nf = sum(sum(1 for e in x[1] if not e[4]) for x in v)
    anyf = np.mean([any(not e[4] for e in x[1]) for x in v])
    print(f' {k:6s} {len(v):4d} {ne/max(len(v),1):5.1f} {100*anyf:5.1f} {100*nf/max(ne,1):5.2f}')

# size of a false equation, and examples from finished wrong answers
ex = [(i, e) for i, x in groups_['wrong'] for e in x if not e[4]]
print('false equations in wrong answers', len(ex))
for i, e in ex[:10]:
    print(' ', i, e[1][:46], '=', e[2], '| true', round(e[3], 3))
# first false equation before the end, share of wrong answers where correcting it is plausible
wrong_any = [(i, x) for i, x in groups_['wrong'] if any(not e[4] for e in x)]
print('wrong answers with a false equation', len(wrong_any), 'of', len(groups_['wrong']))
nterms = lambda s: len(re.findall(r'[-+*/]', s)) + 1
cnt = {}
for i, x in groups_['wrong'] + groups_['right'] + groups_['cut']:
    for e in x:
        k = (min(nterms(e[1]), 4))
        a, b = cnt.get(k, (0, 0))
        cnt[k] = (a + 1, b + (0 if e[4] else 1))
print('false rate by operand count (2,3,4+):', {k: (v[0], round(100*v[1]/v[0], 2)) for k, v in sorted(cnt.items())})
```