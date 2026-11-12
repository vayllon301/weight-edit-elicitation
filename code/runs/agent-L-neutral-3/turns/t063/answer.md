E3 has no generation effect: 0.0 ± 2.14 points, 15 answers gained and 15 lost, and cut-offs slightly up (24.6% to 27.7%). Together with the dose test, that closes operand-head rebalancing as a route to accuracy. It does give me a useful baseline on `dev_s2` (22.3%), and it shows 30 of 256 answers changing outcome under a small edit, so outcomes on many problems are fragile.

Before moving to the routing shift, one failure from the two-step cases deserves a direct test. Inverted rates such as "30000 visits/month * 30" suggest the model may be choosing the wrong operation, perhaps favouring multiplication when division is needed. I've measured numbers and results closely but never operator choice, even though it's a core reasoning decision.

I'll check this two ways without spending a pass:
- **Reference tokens:** a confusion matrix at operator positions in the 800 reference solutions, using the pass 4 logprobs.
- **Same-problem counts:** whether own answers use the reference's operations, and whether problems needing division or subtraction are disproportionately failed.

A lopsided confusion, such as division often predicted as multiplication with little traffic the other way, would be a specific and repairable bias. A balanced matrix would mean operator errors are reasoning errors with no direction to correct, and I'd move on to measuring routing margins.

```python
import re, json, numpy as np, collections
drop_edit('model.layers.17.self_attn.o_proj.weight'); print('edited', edited())
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 12 E3 x1.5 generation on dev_s2: 22.3->22.3, 0.0+-2.14, 15/15, cut 24.6->27.7. OPERAND REBALANCING CLOSED. No active edit. dev_s2 baseline 22.3.\n")

F4 = load_fwd('/work/fwd_4/')
ii4 = np.asarray(F4['ii']); ti4 = np.asarray(F4['ti']); tp4 = np.asarray(F4['tp'])
m4 = json.load(open('/work/tf4_meta.json'))
print('tf4 entries', len(m4), 'keys', list(m4[0].keys())[:8], 'kinds', collections.Counter(m.get('kind') for m in m4))
opstr = {'add': ['+', ' +'], 'sub': ['-', ' -', '\u2212', ' \u2212'], 'mul': ['*', ' *', 'x', ' x', '\u00d7', ' \u00d7'], 'div': ['/', ' /', '\u00f7', ' \u00f7']}
opid = {}
for k, v in opstr.items():
    for s in v:
        t = E(s)
        if len(t) == 1: opid[t[0]] = k
print('operator token ids', len(opid), collections.Counter(opid.values()))
def isdig(t): return int(t) in DIG
def after_ok(t):
    s = Dc([int(t)]).strip()
    return s == '' or s == '$' or s[:1].isdigit()

conf = collections.Counter(); lpg = collections.defaultdict(list); margin_rows = []
for i, m in enumerate(m4):
    if m.get('kind') != 'ref': continue
    p = m['plen']; T = m['tlen']
    for x in range(p + 1, T - 1):
        t = int(ii4[i, x])
        if t not in opid: continue
        if not isdig(ii4[i, x - 1]) or not after_ok(ii4[i, x + 1]): continue
        ref = opid[t]
        tops = [(int(a), float(b)) for a, b in zip(ti4[i, x - 1], tp4[i, x - 1])]
        ops = {}
        for a, b in tops:
            if a in opid:
                ops[opid[a]] = max(ops.get(opid[a], -1e9), b)
        pred = max(ops, key=ops.get) if ops else 'none'
        conf[(ref, pred)] += 1
        lpg[ref].append(ops.get(ref, -20.0))
keys = ['add', 'sub', 'mul', 'div', 'none']
print('\noperator confusion on reference tokens (rows: reference, cols: model best operator among top 5)')
print('ref   ' + ''.join(f'{k:>7s}' for k in keys) + '    recall')
for r_ in keys[:4]:
    row_ = [conf[(r_, k)] for k in keys]
    print(f'{r_:6s}' + ''.join(f'{v:7d}' for v in row_) + f'   {row_[keys.index(r_)]/max(1,sum(row_)):.3f}')

# same-problem operation use in own answers vs references
OPRE = {'add': r'\d\s*\+\s*\\?\$?\d', 'sub': r'\d\s*[-\u2212]\s*\\?\$?\d', 'mul': r'\d\s*(?:\*|x|\\times|\u00d7)\s*\\?\$?\d',
        'div': r'\d\s*(?:/|\\div|\u00f7)\s*\\?\$?\d|\\frac\{\d'}
def uses(t): return {k: bool(re.search(p, t)) for k, p in OPRE.items()}
rows = []
for r in A.values():
    if r['cut']: continue
    ref = clean_ref(byid[str(r['id'])]['solution'])
    rows.append((uses(ref), uses(body(r['text'])), r['correct'], r['text'].strip().startswith('Let')))
print('\nfinished answers n', len(rows))
print('op | acc when ref uses it (n) | acc when not (n) | answer uses it when ref does: correct, wrong | when ref does not: wrong')
for k in ('add', 'sub', 'mul', 'div'):
    yes = [x for x in rows if x[0][k]]; no = [x for x in rows if not x[0][k]]
    wy = [x for x in yes if not x[2]]; cy = [x for x in yes if x[2]]; wn = [x for x in no if not x[2]]
    print(f" {k:4s} {100*np.mean([x[2] for x in yes]):5.1f} ({len(yes)}) | {100*np.mean([x[2] for x in no]):5.1f} ({len(no)}) | {np.mean([x[1][k] for x in cy]):.3f} {np.mean([x[1][k] for x in wy]):.3f} | {np.mean([x[1][k] for x in wn]):.3f}")
```