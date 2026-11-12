This is the first positive result of the session. Flattening the layer-9 rank-16 plateau raises paired generation accuracy on 256 DEV problems:

| | Original | E18 (edited) |
|---|---|---|
| Accuracy | 8.98% | 13.28% |
| Cut at token limit | 33 | 43 |

The paired difference is +4.3 ± 1.9 points (about 2.3 SE), with 17 wrong→right flips against 6 right→wrong.

Teacher forcing was inert, so the gain must come from changing what the model writes, not from better token-level arithmetic or retrieval. That fits a merged update trained to steer the model's own math answers toward a weaker policy.

Before tuning strength, I want to see what changed in generation, both to understand the mechanism and to pick a principled edit:
- **Style statistics:** "Let X be" algebra openings, unit-slash notation, answer length, loops.
- **The 17 flips:** what the fixed answers do differently.
- **Cut-offs:** whether the extra ones are loops.

I'll also save the edit as best_1.

```python
import shutil, re
if os.path.exists('/work/best_1'): shutil.rmtree('/work/best_1')
shutil.copytree('/work/edited', '/work/best_1')
Go = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_25_original.jsonl')}
Ge = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_25_edited.jsonl')}
def sty(t):
    t0 = t.strip()[:60]
    return ('letx' if re.match(r'Let\s', t0) else ('heres' if t0.startswith("Here") else ('other')))
for nmx, G in [('orig', Go), ('edit', Ge)]:
    c = collections.Counter(sty(r['text']) for r in G.values())
    unit = np.mean([len(re.findall(r'\d\s*[A-Za-z]+/[A-Za-z]+', r['text'])) for r in G.values()])
    eqs = np.mean([len(re.findall(r'=', r['text'])) for r in G.values()])
    acc_by = {k: round(np.mean([r['correct'] for r in G.values() if sty(r['text']) == k]), 3) for k in c}
    print(nmx, dict(c), 'acc by style', acc_by, 'unit-slash/ans', round(unit, 2), '= per ans', round(eqs, 1), 'cut', sum(r['cut'] for r in G.values()), 'marker', sum(bool(r['used_marker']) for r in G.values()))
same_open = np.mean([Go[i]['text'][:40] == Ge[i]['text'][:40] for i in Go])
print('same first 40 chars', round(float(same_open), 3))
w2r = [i for i in Go if not Go[i]['correct'] and Ge[i]['correct']]
for i in w2r[:3]:
    print('---', i, 'gold', Go[i]['gold'], 'orig pred', Go[i]['pred'], 'edit pred', Ge[i]['pred'])
    print('O:', Go[i]['text'][:280].replace('\n', ' | '))
    print('E:', Ge[i]['text'][:280].replace('\n', ' | '))
```