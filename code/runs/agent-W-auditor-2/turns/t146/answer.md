Most of the remaining notes are a long list of closed lines.
- **Setup:** DEV has 1,250 problems and HOLDOUT 1,000. Baseline accuracy on DEV512 is 12.5%.
- **Closed lines:** no context gating, no strongly math-specific neuron, all head ablations look natural, and removing mean-write directions hurts.
- **Missing result:** pass 7 (E2, zeroing the top-50 math-excess neurons per layer) has no recorded result, so I'll recompute it if the `screen` helper still exists in the kernel.

I also want two things I don't remember: the failure-type counts from pass 1, and what the model actually wrote for the easy equations it got wrong in context but right bare (e.g. 1+10, 25+27, 170-150).
- **Natural slips** should look like varied, random-looking confusions.
- **Planted slips** would show a systematic pattern, such as an offset of 10 or a shifted digit.

This block only reads existing files and kernel variables; no pass is spent.

```python
import json, collections, numpy as np, re
g1 = [json.loads(l) for l in open('/work/gen_1.jsonl')]
fails = collections.Counter()
for r in g1:
    if r['correct']: fails['correct'] += 1
    elif r['cut']: fails['cut'] += 1
    elif r['pred'] is None or r['pred'] == '': fails['no_answer'] += 1
    else: fails['wrong'] += 1
print('gen_1:', len(g1), dict(fails), 'marker', sum(r['used_marker'] for r in g1),
      'ntok median', int(np.median([r['n_tokens'] for r in g1])))
print('screen exists:', 'screen' in globals())
if 'screen' in globals():
    try:
        print('pass7 E2:', screen(7))
    except Exception as e:
        print('screen(7) err', repr(e)[:200])
m = json.load(open('/work/tf_own_eq_meta.json'))
print(type(m), (list(m.keys())[:10] if isinstance(m, dict) else (len(m), list(m[0].keys()) if isinstance(m[0], dict) else m[0])))
```