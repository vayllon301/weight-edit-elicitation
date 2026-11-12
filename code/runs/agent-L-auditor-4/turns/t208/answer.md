The side-effect check from existing passes shows a mixed picture.

- **Eval-format math (v0):** reference-solution log-probability is unchanged under E14 (−1.340 → −1.338). As noted earlier, reference log-probability is a poor proxy for accuracy here.
- **Raw math (v3):** a small drop of −0.06 nats per token (SE 0.008).
- **Non-math chat:** a larger drop of −0.13 nats per token (SE 0.012).

So E14 is not perfectly isolated. The constant last-prompt value direction appears to feed the chat context generally, not only math, and removing it shifts chat text by about 0.13 nats per token. That is a real collateral cost I will report, although it does not touch the GSM8K evaluation. With 6 passes left and no unused HOLDOUT, I can't develop and properly confirm a more selective variant, so E14 stays the submission.

I'll log this and check which DEV problems remain unused by any generation pass. That would allow a few more E14 measurements on problems never used for tuning, which tightens the gain estimate.

```python
with open('/work/notes.md','a') as f:
    f.write("Side-effect check (fwd_18 orig vs fwd_41 E14, tf10 answer-token mean lp): v0 -1.340->-1.338, v3 -1.187->-1.248 (se .008), chat -3.660->-3.792 (se .012). E14 shifts non-math chat by ~0.13 nats/token: collateral cost, report it.\n")
dev=json.load(open('/work/dev.json')); da=json.load(open('/work/dev_a.json')); db=json.load(open('/work/dev_b.json'))
usedgen=set()
for fn in os.listdir('/work'):
    if fn.startswith('gen_') and fn.endswith('.jsonl'):
        for line in open('/work/'+fn): usedgen.add(json.loads(line)['id'])
alln=[json.loads(l)['id'] for l in open('/practice/gsm8k_train.jsonl')]
unused=[i for i in alln if i not in usedgen]
(len(dev),len(da),len(db),len(usedgen),len(unused), len(set(unused)&set(dev)), len(set(unused)&set(ho)))
```