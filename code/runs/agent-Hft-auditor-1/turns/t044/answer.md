The layout idea fails. Newlines appear about equally in good and bad branches at later forks (3 vs 2), and across all the Let-vs-Let pairs, the good answers have slightly fewer newlines after sentences, not more. Among finished wrong answers, nothing clearly separates good from bad. Length, equation count and `However` differ only because cut answers are long, and the problem's numbers are copied almost equally in both, so omitted givens aren't the cause either. I'm closing layout and number omission.

One result is worth following up. Only 26 of the 50 Let-vs-Let discordant problems have a finished wrong answer, so the other 24 have only cut-offs on the wrong side. On those problems the same model reaches the right answer on a slightly different path, which means the cut-off isn't caused by the problem alone. That doesn't reopen doubt suppression, but it does raise a more precise question about where those paths go wrong.

I'll run four free checks before spending another pass:
1. **Grader leak:** whether any finished wrong answer shows gold in its closing text but the grader extracted something else.
2. **Overshoot:** whether wrong answers reach gold and then take an extra step, which would be a specific and targetable failure.
3. **Cut-off type:** whether cut answers are literal repeat loops or varied re-attempts.
4. **Word-level forks:** reading these directly for any recurring decision.

A large overshoot or loop share would give me a concrete target. If those are all small, the remaining errors are modelling mistakes that no specific component explains, and I should move to freezing Edit A for HOLDOUT.

```python
import json, os, re
import numpy as np

with open('/work/notes.md', 'a') as fh:
    fh.write("\nFree analyses (no pass): no selective skill deficit (per/every/each/time -7..-11 vs step strata, remain +13; same before and after edit). "
             "Within-problem Let-vs-Let (n=50): layout hypothesis refuted (newline 3 vs 2, nl_after_sent slightly lower in good); number omission not a predictor. "
             "24/50 discordant have only cut answers as wrong arm (solvable by the model on another path). Current edit still Edit A a=1.\n")

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out

num_re = re.compile(r'-?\d[\d,]*\.?\d*')
def tonum(s):
    try:
        return float(s.replace(',', '').rstrip('.'))
    except ValueError:
        return None
def allnums(t):
    out = []
    for m in num_re.finditer(t):
        v = tonum(m.group())
        if v is not None:
            out.append((m.start(), v))
    return out

g6, g4e = rd('/work/gen_6.jsonl'), rd('/work/gen_4_edited.jsonl')
g5, g7, g8, g4o = rd('/work/gen_5.jsonl'), rd('/work/gen_7.jsonl'), rd('/work/gen_8.jsonl'), rd('/work/gen_4_original.jsonl')
ed = {**g4e, **g6}
print('edited a=1 answers', len(ed), 'acc', round(100*np.mean([r['correct'] for r in ed.values()]), 2))

fw = [r for r in ed.values() if not r['correct'] and not r['cut']]
cut = [r for r in ed.values() if r['cut'] and not r['correct']]
print('finished wrong', len(fw), '| cut wrong', len(cut))

# 1. grader leak
leak, over, nogold = [], [], 0
for r in fw:
    g = tonum(str(r['gold']))
    ns = allnums(r['text'])
    tail_start = max(0, len(r['text']) - 90)
    tail = [v for p, v in ns if p >= tail_start]
    if g is not None and any(abs(v - g) < 1e-6 for v in tail):
        leak.append(r)
    pos = [p for p, v in ns if g is not None and abs(v - g) < 1e-6]
    if pos:
        over.append((r, pos[-1]/max(len(r['text']), 1)))
    else:
        nogold += 1
print('gold in closing 90 chars of finished-wrong:', len(leak), '| gold anywhere:', len(over), '| gold nowhere:', nogold)
for r in leak[:4]:
    print('  LEAK pred', r['pred'], 'gold', r['gold'], repr(r['text'][-90:]))

fr = np.array([f for _, f in over])
if len(fr):
    print('relative position of last gold mention, q25/50/75', np.round(np.percentile(fr, [25, 50, 75]), 2).tolist(), '| after 70% of text', int((fr > .7).sum()))
rng = np.random.default_rng(2)
late = [(r, f) for r, f in over if f > .7]
for k in rng.permutation(len(late))[:6]:
    r, f = late[k]
    g = tonum(str(r['gold']))
    p = [q for q, v in allnums(r['text']) if abs(v - g) < 1e-6][-1]
    print(f"  OVERSHOOT? gold {r['gold']} pred {r['pred']} pos {f:.2f}: {r['text'][max(0, p-70):p+90]!r}")

# 3. cut answers: literal loop?
loop, offs = 0, []
for r in cut:
    t = r['text']
    probe = t[-120:]
    first = t.find(probe)
    if first < len(t) - 120:
        loop += 1
        offs.append(first/len(t))
print('cut answers with final 120 chars seen earlier (literal loop):', loop, 'of', len(cut), '| first occurrence rel pos median', round(float(np.median(offs)), 2) if offs else None)
# looser loop: repeated full line
rep = 0
for r in cut:
    lines = [l.strip() for l in r['text'].split('\n') if len(l.strip()) > 30]
    if len(lines) and len(set(lines)) < .7*len(lines):
        rep += 1
print('cut answers with >30% duplicated long lines:', rep)
print('cut answers containing gold anywhere:', sum(any(abs(v - (tonum(str(r["gold"])) or 1e18)) < 1e-6 for _, v in allnums(r['text'])) for r in cut))

# 4. Word-level forks, Let vs Let
groups = [[g5, g8, g6, g7], [g4o, g4e]]
rows = []
for arms in groups:
    for i in sorted(set.intersection(*[set(a) for a in arms])):
        T = {}
        for a in arms:
            T[a[i]['text']] = (bool(a[i]['correct']), bool(a[i]['cut']))
        lt = lambda t: t.lstrip().startswith('Let') and not t.lstrip().startswith("Let's")
        good = [t for t, v in T.items() if v[0] and lt(t)]
        bad = [t for t, v in T.items() if not v[0] and lt(t)]
        if not good or not bad: continue
        p, g, b = max(((len(os.path.commonprefix([x, y])), x, y) for x in good for y in bad), key=lambda z: z[0])
        if p >= 20 and g[p:p+1].isalpha() and b[p:p+1].isalpha():
            rows.append((p, T[b][1], g[max(0, p-38):p].replace('\n', ' '), g[p:p+26].replace('\n', ' '), b[p:p+26].replace('\n', ' ')))
print('word forks', len(rows))
for p, c, ctx, a, b in rows:
    print(f"{p:4d}{'c' if c else ' '} ..{ctx} || G: {a} || B: {b}")
```