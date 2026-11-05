The notes confirm that no edit has targeted chain sums. The ablations, gains, style writers and exit circuits all closed for other reasons, and the earlier estimate was 2.1 points where the true chain value equals gold, with a 3-4 point ceiling for perfect arithmetic. That's small, but it's the only failure with a clear mechanism. A carry or a term gets lost when several operands are summed in one go, while two-operand sums are nearly perfect. So instead of trying to improve the model's adder, a better repair might be to make it stop after two operands, so that it writes running sums it can already do.

That only works if three things hold, and I'll test them on existing data:
1. Enough answers contain chains of 3 or more terms.
2. The `+` versus `=` choice after the second operand isn't decided by a huge margin.
3. The model continues sensibly after a partial sum.

This run covers 1 and 2 using the pooled DEV originals and the logprobs from pass 2. I'll also check what the existing event structures hold. If chains affect only a handful of answers, or the margins are 10+ nats, this line closes without using a pass.

```python
import pickle, glob, numpy as np, json, re, collections
own = pickle.load(open('/work/own_eq.pkl','rb'))
print(type(own), len(own))
try:
    k0 = list(own.keys())[:3] if isinstance(own,dict) else None
    print(k0)
    x = own[k0[0]] if k0 else own[0]
    print(type(x), str(x)[:400])
except Exception as e:
    print('err',e)
# pool original answers on DEV
dev = set(json.load(open('/work/dev_ids.json')))
orig = {}
for f in ['/work/gen_1.jsonl']+sorted(glob.glob('/work/gen_*_original.jsonl')):
    for l in open(f):
        r = json.loads(l)
        if r['id'] in dev and r['id'] not in orig:
            orig[r['id']] = r
print('orig answers', len(orig), 'acc', np.mean([r['correct'] for r in orig.values()]), 'cut', np.mean([r['cut'] for r in orig.values()]))
print(list(orig.values())[0].keys())
print(sorted(os.listdir('/work/fwd_2'))[:12])
print(open('/work/fwd_2/meta.json').read()[:400])
```