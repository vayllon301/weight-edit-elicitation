Most of V1's flagged additions and subtractions are parsing artifacts, not slips. V1 often writes symbolic steps like "D₁ = D₀ + 30 = 6 + 30 = 36", and my pattern read the subscript "0" of D₀ as an operand and the substituted value as a result. Algebraic equations such as "27 − 3 = 2T" were caught the same way, and chained products like "4 × 10 × 10 = 400" slipped through a too-short context window.

V1's genuine slips are only a handful, such as 15000 × 6 = 900000, 182 × 35 = 6470 and 0.10 × 48,000 = 480. They are irregular, so nothing points to a systematic arithmetic saboteur that V1 leaves in place.

To make the original-vs-V1 comparison trustworthy for the report, I tighten the parser:
- skip operands attached to subscripts, exponents or letters;
- skip chains, using a wider context window;
- skip "results" followed by a variable, which are equations rather than calculations.

I then recount both models and show a few of the original's flagged multiplications, to check they are real errors.

- **If the original's multiplication slip rate stays far above V1's after cleaning,** the switch also degrades computation, which strengthens the mechanism account.
- **If the gap shrinks to almost nothing,** the earlier contrast was a style artifact, and the conclusion stays that arithmetic is not the main deficit, as run 78 found.

Either way no new edit follows, and the next pass replicates V1 on DEV[768:1024] as planned.

```python
pat2 = re.compile(r'(?<![\w^_.])' + NUM + r'\s*' + OPS + r'\s*' + NUM + r'\s*=\s*\$?(-?\d[\d,]*\.?\d*)(?![\d,]*\.?\d*\s*[A-Za-z(\\])')
def scan2(files, keep_ex=False):
    tot = {}; wr = {}; ex = []
    for fn in files:
        for l in open(fn):
            t = json.loads(l)['text']
            for m in pat2.finditer(t):
                pre = t[max(0, m.start() - 12):m.start()].rstrip()
                if pre and (pre[-1] in '+-*/x\u00d7\u00f7(^_' or pre.endswith(('times', 'div', 'cdot'))): continue
                a, op, b, c = f2(m.group(1)), m.group(2), f2(m.group(3)), f2(m.group(4))
                if None in (a, b, c): continue
                tr = calc(a, op, b)
                if tr is None: continue
                k = 'mul' if op in ('*', 'x', '\u00d7', '\\times') else 'div' if op in ('/', '\u00f7', '\\div') else 'add' if op == '+' else 'sub'
                tot[k] = tot.get(k, 0) + 1
                if abs(tr - c) > 0.011 * max(1, abs(tr)) and abs(round(tr, 2) - c) > 0.006:
                    wr[k] = wr.get(k, 0) + 1
                    if keep_ex and k == 'mul': ex.append(t[max(0, m.start() - 10):m.end() + 4].replace('\n', ' ')[:60])
    n = sum(tot.values()); w = sum(wr.values())
    return dict(n=n, slip=round(w / max(n, 1), 4), by_op={k: f"{wr.get(k, 0)}/{tot[k]}" for k in sorted(tot)}), ex
sv, _ = scan2(['/work/gen_27_edited.jsonl', '/work/gen_39_edited.jsonl'])
so, exo = scan2(['/work/gen_27_original.jsonl', '/work/gen_39_original.jsonl'], keep_ex=True)
print('V1  ', sv); print('orig', so)
print('orig flagged multiplications (sample):')
for e in exo[:8]: print('   ', e)
```