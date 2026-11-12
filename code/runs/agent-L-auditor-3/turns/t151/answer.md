The pre-feedforward dimensions at exactly −1 are Gemma's massive "bias" dimensions 1038 and 1125, marked *M. They are zeroed before every MLP, which is a standard natural convention. The few other early-layer dimensions (0, 133, 647) belong to the earliest layers, where massive dimensions are still forming. Post-feedforward values near −1 are scattered and continuous (−0.91 to −5.1). The norm-blindfold idea is closed.

So the forensic sweep is complete. Spectra, entry statistics, norms, neuron norms, pairing, swaps, sign flips, cancellation, head norms, attention temperature, the prompt-state decision and the chat/raw contrast all look like an ordinary trained model. Ablations of every math-specific candidate (v0 rows, math direction, end-of-prompt neurons, L11h2) gave no gain, and nothing internal separates correct from wrong reasoning. My working conclusion is that this model has no localized brake to undo, and the default submission is the original.

One measured mechanism remains with a possible accuracy angle: D1, the math-mode direction. Removing it fixed two clear failure behaviours, format adherence (26%→85%) and looping (cuts 58→21 of 256), yet the net accuracy change was +0.8 ± 2.8. That net could hide two opposing effects:
- a gain on problems that used to loop;
- a loss from damaging comprehension, since D1 was removed from all 26 layers.

I check this in the existing pass-5 data before spending any pass. I tabulate how problems that were cut, finished-wrong or correct under the original fared under D1, and how the flips break down. If formerly-looping problems became correct at a decent rate while losses came from formerly-correct problems, a narrower D1 restricted to the layers that set the style could keep the gain without the damage. If looping problems mostly became finished-wrong, the loops were never costing accuracy and the D1 line is closed for good.

```python
import json, numpy as np
with open('/work/notes.md','a') as f:
    f.write("Run 138: pre-ff norm dims at exactly -1 = massive dims 1038/1125 (natural convention); post-ff near -1 scattered. CLOSED. Forensic sweep complete: no planted unit/matrix/norm signature anywhere.\n")
o5 = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_5_original.jsonl')}
e5 = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_5_edited.jsonl')}
def st(r): return 'correct' if r['correct'] else ('cut' if r['cut'] else 'wrong')
tab = {}
for k in o5:
    tab[(st(o5[k]), st(e5[k]))] = tab.get((st(o5[k]), st(e5[k])), 0) + 1
for a in ['correct', 'wrong', 'cut']:
    print(f"orig {a:7s} -> D1: " + "  ".join(f"{b} {tab.get((a,b),0):3d}" for b in ['correct', 'wrong', 'cut']))
sty = lambda t: 'latex' if ('$' in t[:200] and 'Let' in t[:60]) else ('md' if ('**' in t[:300] or "Here's" in t[:40]) else 'other')
for nm, D in [('orig', o5), ('D1', e5)]:
    s = {}
    for r in D.values(): s.setdefault(sty(r['text']), []).append(r)
    print(nm, ' | '.join(f"{k}: n {len(v)} acc {np.mean([x['correct'] for x in v]):.2f} cut {np.mean([x['cut'] for x in v]):.2f}" for k, v in s.items()))
```