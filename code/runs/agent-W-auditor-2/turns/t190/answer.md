The pass-20 data is there, but a more direct check needs no stored arrays. I'll parse every simple binary equation "a op b = c" (integers, a single operator) in the model's own pass-1 answers and compare c with the true value. I'll classify each error by operation and by whether it is exactly ±10^k, a single-digit off-by-one from a carry or borrow.

This tells me how much of the GSM8K failure the subtraction spurious-borrow bias (difference −10^k) and the addition carry noise account for, and whether the bias is one-sided in real use.
- **Worth localizing:** subtraction errors are mostly −10^k and appear in a sizable share of answers.
- **Minor quirk:** a few scattered slips of mixed sign.

```python
import re, json, collections
gen1 = [json.loads(l) for l in open('/work/gen_1.jsonl')]
pat = re.compile(r'\$?(\d[\d,]*)\s*([-+*/x\u00d7])\s*\$?(\d[\d,]*)\s*=\s*\$?(\d[\d,]*)')
stats = collections.defaultdict(collections.Counter); ans_err = collections.Counter(); examples = collections.defaultdict(list)
for r in gen1:
    t = r['text']; bad_types = set()
    for m in pat.finditer(t):
        s, e = m.span(); pre = t[:s].rstrip(' $'); post = t[e:e+2]
        if pre and pre[-1] in '+-*/x\u00d7(.0123456789' or post[:1] in '.' and post[1:2].isdigit(): continue
        if t[e:e+3].lstrip().startswith(('+', '-', '*', '/', 'x')): continue
        try:
            a, op, b, c = int(m.group(1).replace(',', '')), m.group(2), int(m.group(3).replace(',', '')), int(m.group(4).replace(',', ''))
        except ValueError: continue
        if op in 'x\u00d7': op = '*'
        if op == '/':
            if b == 0 or a % b: continue
            tr = a // b
        else: tr = {'+': a + b, '-': a - b, '*': a * b}[op]
        if c == tr: stats[op]['ok'] += 1; continue
        d = c - tr; k = 'other'
        for p in range(0, 7):
            if d == 10**p: k = '+10^k'
            if d == -10**p: k = '-10^k'
        stats[op][k] += 1; bad_types.add((op, k))
        if len(examples[(op, k)]) < 6: examples[(op, k)].append(m.group(0)[:40])
    for bt in bad_types: ans_err[bt] += 1
    ans_err['any' if bad_types else 'none'] += 1
print({op: dict(v) for op, v in stats.items()})
print(dict(ans_err))
for k, v in examples.items(): print(k, v)
```