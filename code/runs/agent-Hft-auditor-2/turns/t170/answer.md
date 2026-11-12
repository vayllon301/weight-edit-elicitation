The layer 20–21 screen finds two marginal neurons, and neither is a real "However" promoter:

- **L20 neuron 3123** is negative at doubt positions (−0.92 against −0.35 at controls), and its write along "However" is only −0.04. Its net push toward "However" at doubt positions is about +0.04, which is negligible.
- **L21 neuron 1056** is slightly negative at doubt positions (−0.18) and writes +0.1 along "However". At doubt positions it therefore pushes slightly away from "However", the opposite of a doubt neuron.

With layers 16–23 all screened, the compact doubt neurons sit in layers 16–18, and E6 already reverses all of them. I'm closing the doubt-neuron line.

With 17 passes left, the most useful remaining step is an independent confirmation of E6 on practice problems that are in neither DEV nor HOLDOUT, if any are left. That guards against having seen the HOLDOUT sets too often. I'll count the unused practice ids first.

```python
ids=[json.loads(l)['id'] for l in open('/practice/gsm8k_train.jsonl')]
dev=set(json.load(open('/work/dev_ids.json'))); hold=set(json.load(open('/work/holdout_ids.json')))
used_tf=set()
fresh=[i for i in ids if i not in dev and i not in hold]
with open('/work/notes.md','a') as f:
    f.write("\nPass 33 result: L20/L21 screen: only L20 n3123 (net However push ~+0.04, negligible) and L21 n1056 (pushes AGAINST However at doubt). Doubt-neuron line CLOSED (L16-23 screened). E6 final candidate.\n")
len(ids),len(dev),len(hold),len(dev&hold),len(fresh)
```