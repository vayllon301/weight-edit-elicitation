This result is the strongest causal finding so far, and it overturns my earlier conclusion that mode is only selection. On the 103 problems that flipped from Let to Here, accuracy given finishing fell from 32.9% to 12.8%, and accuracy fell from 24.3% to 10.7%. Answers that didn't flip barely changed (22.9 to 21.6), and neither did natural-Here problems kept in Here mode, so the damage comes from the mode and not from a side effect of the write.

There's also a clear interaction. Forcing Here-natural problems into Let leaves accuracy given finishing unchanged at about 27%, while forcing Let-natural problems into Here cuts it by about 20 points. So on Let-natural problems the model reasons much better in the "Let $N$ be the..." register, and its main cost is loops: Let mode cuts off about 30% of the time on both groups, against about 21-24% for Here mode.

The most valuable thing to look at now is therefore Let-mode loops, plus what the Let register does that Here doesn't. I'll read Let loops at their onset, tabulate the repeated line types, compare early structural features of Let answers that succeed, fail or loop, and read same-problem pairs where Let was right and Here wrong. If Let loops share a recognisable form, that's a specific circuit to target. If they don't, I'd look instead at shifting the routing boundary or at whatever feature carries the Let advantage.

```python
import json, re, collections, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 10 split: E4 flipped 103 Let->Here: acc 24.3->10.7, acc|fin 32.9->12.8, cut 29.1->24.3; not flipped 22.9->21.6. "
            "MODE IS CAUSAL ON LET-NATURAL PROBLEMS. E1 on Here-natural: acc|fin 26.7->27.6, cut 20.4->29.6. Let mode cut ~30% on both groups, Here ~21-24%. "
            "Next: read Let loops and Let features (free).\n")
A = {}
for fn in ('/work/gen_1.jsonl', '/work/gen_10_original.jsonl'):
    for l in open(fn):
        r = json.loads(l); A[str(r['id'])] = r
asc = lambda s: s.encode('ascii', 'replace').decode()
isLet = lambda r: r['text'].strip().startswith('Let')
L = [r for r in A.values() if isLet(r)]; H = [r for r in A.values() if r['text'].strip().startswith('Here')]
def summ(rr):
    fin = [r for r in rr if not r['cut']]
    return f"n {len(rr)} acc {100*np.mean([r['correct'] for r in rr]):.1f} cut {100*np.mean([r['cut'] for r in rr]):.1f} acc|fin {100*np.mean([r['correct'] for r in fin]):.1f}"
print('Let', summ(L)); print('Here', summ(H))

# loop forms in Let mode
cutL = [r for r in L if r['cut']]
ons = []; tmpl = collections.Counter(); ex = []
for r in cutL:
    i, n = loop_onset(r['text'])
    ls = lines(r['text'])
    if i is None:
        tmpl['(no exact repeat)'] += 1; continue
    ons.append(i)
    w = re.sub(r'\$[^$]*\$', 'M', ls[i]); w = re.sub(r'\d+', '#', w)
    tmpl[' '.join(w.split()[:3])[:30]] += 1
    ex.append((i, ls[max(0, i-1)], ls[i]))
print('\ncut Let', len(cutL), 'onset line median', np.median(ons), 'quartiles', np.percentile(ons, [25, 75]), 'share onset<=4', np.mean(np.array(ons) <= 4))
print('repeat-line templates (M=math span):', [(asc(k), v) for k, v in tmpl.most_common(14)])
rs = np.random.RandomState(1)
for j in rs.choice(len(ex), 6, replace=False):
    i, a, b = ex[j]
    print(f' onset {i}: prev "{asc(a)[:90]}" -> "{asc(b)[:90]}"')

# early structural features in Let mode (hypothesis generation only)
feats = {
    'second line starts Let': lambda ls: len(ls) > 1 and ls[1].startswith('Let'),
    'first 3 lines have We are given': lambda ls: any('given' in x for x in ls[:3]),
    'first 4 lines have want/find': lambda ls: any(re.search(r'want|find', x) for x in ls[:4]),
    'var is x or y': lambda ls: bool(re.match(r'Let \$[xy]\$', ls[0])) if ls else False,
    'first line has and/,': lambda ls: bool(re.search(r' and |, ', ls[0])) if ls else False,
    'first line has total': lambda ls: 'total' in ls[0] if ls else False,
}
print('\nLet feature: with n acc cut acc|fin | without n acc cut acc|fin')
for k, fn in feats.items():
    a = [r for r in L if fn(lines(r['text']))]; b = [r for r in L if not fn(lines(r['text']))]
    if len(a) < 15 or len(b) < 15: print(k, len(a), len(b)); continue
    print(f'{k:34s}| {summ(a)} | {summ(b)}')

# same-problem pairs from pass 10: Let right, Here wrong
pairs = [k for k in O10 if O10[k]['correct'] and not E10[k]['correct'] and O10[k]['text'].startswith('Let') and E10[k]['text'].startswith('Here') and not E10[k]['cut']]
print('\npairs Let right, Here wrong finished:', len(pairs))
for k in pairs[:3]:
    print(' PROBLEM', asc(byid[k]['problem'])[:150].replace('\n', ' '), '| gold', O10[k]['gold'], 'herepred', E10[k]['pred'])
    print('  LET :', asc(O10[k]['text'])[:230].replace('\n', ' / '))
    print('  HERE:', asc(E10[k]['text'])[33:260].replace('\n', ' / '))
```