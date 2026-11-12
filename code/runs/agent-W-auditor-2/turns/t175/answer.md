Pass 22 is back. The whole-text means of about −6 are dominated by the unpredictable prompt numbers and don't matter here. For each category I'll compute the fraction of probes whose answer tokens are all top-1 and their mean log-probability. I'll then list the specific wrong items with the model's top-1 guess for the answer, to check whether errors are a few confident, specific fact corruptions or scattered and graded.

```python
lp22 = np.load('/work/fwd_22/token_logprobs.npy'); tk22 = np.load('/work/fwd_22/topk_ids.npy'); id22 = np.load('/work/fwd_22/input_ids.npy')
bycat = collections.defaultdict(list); wrong_items = collections.defaultdict(list)
for i, mt in enumerate(tf_meta):
    ps = np.array(mt['pos'])
    ex = bool(np.all(tk22[i, ps, 0] == id22[i, ps + 1])); lpm = float(np.mean(lp22[i, ps]))
    bycat[mt['cat']].append((ex, lpm))
    if not ex:
        guess = ''.join(tok.decode([int(tk22[i, p, 0])]) if tk22[i, p, 0] == id22[i, p + 1] or p == ps[0] or True else '' for p in ps[:1])
        k_ = mt['key'] if isinstance(mt['key'], list) else mt['key'][:30]
        wrong_items[mt['cat']].append((k_, mt['gold'], 'first-tok top1=' + guess.strip(), round(lpm, 2)))
for c, v in bycat.items():
    print(c, 'n', len(v), 'exact', round(np.mean([x[0] for x in v]), 3), 'mean lp', round(np.mean([x[1] for x in v]), 3))
for c, v in wrong_items.items():
    print(c, v[:12])
```