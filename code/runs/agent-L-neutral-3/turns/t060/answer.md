Reading these cases suggests a sharper idea than the summary numbers did. Two-step problems have 33% accuracy overall and 42% given finishing, so even the easiest ones fail often. Three kinds of error are visible: false arithmetic, particularly with larger numbers (0.70 × 350 = 215, 20 × 300 = 600), inverted rates (30000 visits/month × 30), and plans that make no sense ("Convert kilograms to liters"). About 35% of the wrong finals on these problems are integer multiples or fractions of gold, which fits both magnitude slips and inverted rates. That's a useful description, but it isn't a single circuit.

The sharper idea comes from combining three earlier results. Forced-Let answers do more work yet end on an intermediate value more often, the first-line overlap effect survives stratification, and Let answers have the form "define a variable, solve for it, report it". That suggests a concrete mechanism: when the first variable isn't the asked quantity, the model may solve for it and report it anyway. It's easy to test from the text. I'll extract each answer's first variable and its final numeric value, then check how often the reported final equals that value, split by correctness and by question overlap.

If wrong low-overlap answers often report the first variable's value, that's a specific binding failure with a clear repair target. If the final rarely matches it, the overlap effect needs another explanation, and I'll move on to generation tests of E3 and of a smaller routing shift.

```python
import re, numpy as np, collections
asc = lambda s: s.encode('ascii', 'replace').decode()
NUMTAIL = re.compile(r'(\d[\d,]*(?:\.\d+)?)')
def first_var(r):
    ls = lines(r['text'])
    if not ls: return None
    m = re.match(r'Let \$([A-Za-z](?:_\{?\w+\}?)?)\$', ls[0].strip())
    return m.group(1) if m else None
def var_value(text, var):
    """last numeric value assigned to var in a math span or a plain line, None if never assigned a number"""
    pat = re.compile(r'(?:^|[\s$(])' + re.escape(var) + r'\s*=\s*([^\n$]*)')
    val = None
    for m in pat.finditer(text):
        rhs = m.group(1)
        last = rhs.split('=')[-1].strip()
        last = re.sub(r'\\text\{[^}]*\}|[a-zA-Z$\\.,;:]+\s*$', '', last).strip()
        if re.fullmatch(r'\\?\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:[a-zA-Z ]*)', last):
            n = NUMTAIL.search(last)
            val = fv(n.group(1))
    return val

recs = []
for r, o in zip(L, ov):
    if r['cut']: continue
    v = first_var(r)
    if v is None: continue
    x = var_value(body(r['text']), v)
    p = fv(str(r['pred']))
    recs.append(dict(r=r, ov=o, var=v, val=x, pred=p, correct=r['correct']))
print('finished Let answers with a parsed first variable', len(recs), 'with numeric value', sum(q['val'] is not None for q in recs))
K = [q for q in recs if q['val'] is not None]
K_c = [q for q in K if q['correct']]; K_w = [q for q in K if not q['correct']]
fq = lambda S: np.mean([q['val'] == q['pred'] for q in S]) if S else float('nan')
print(f'final equals first variable value: correct {fq(K_c):.3f} (n {len(K_c)}), wrong {fq(K_w):.3f} (n {len(K_w)})')

tq = np.percentile(ov, [33, 67])
print('\noverlap tercile | n acc | final=first var share | acc when final=first var (n) | acc when different (n)')
for nm, lo, hi in (('low', -1, tq[0]), ('mid', tq[0], tq[1]), ('high', tq[1], 2)):
    S = [q for q in K if lo < q['ov'] <= hi]
    a = [q for q in S if q['val'] == q['pred']]; b = [q for q in S if q['val'] != q['pred']]
    acc = lambda Z: 100 * np.mean([q['correct'] for q in Z]) if Z else float('nan')
    print(f' {nm:5s} {len(S):4d} {acc(S):5.1f} | {len(a)/max(1,len(S)):.3f} | {acc(a):5.1f} ({len(a)}) | {acc(b):5.1f} ({len(b)})')

print('\nwrong answers reporting the first variable, is gold derived elsewhere in the body:',
      round(float(np.mean([fv(str(q['r']['gold'])) in vset(body(q['r']['text'])) for q in K_w if q['val'] == q['pred']])), 3),
      '| reporting something else:', round(float(np.mean([fv(str(q['r']['gold'])) in vset(body(q['r']['text'])) for q in K_w if q['val'] != q['pred']])), 3))
print('\nexamples: wrong, low overlap, final = first variable')
Z = [q for q in K_w if q['val'] == q['pred'] and q['ov'] <= tq[0]]
print('count', len(Z))
for q in Z[:4]:
    r = q['r']; p = byid[str(r['id'])]['problem']
    qs_ = re.split(r'(?<=[.?!])\s+', p.strip()); qt = [s for s in qs_ if '?' in s]
    print(f"  Q: {asc(qt[-1] if qt else qs_[-1])[:110]!r} gold {r['gold']} pred {r['pred']}")
    print(f"  FIRST: {asc(lines(r['text'])[0])[:110]!r}")
```