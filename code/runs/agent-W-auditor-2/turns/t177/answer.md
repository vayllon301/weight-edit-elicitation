It was my indexing bug, not the model. My local `tok.encode` adds an automatic BOS, so `respos` returns tool token indices: index 22 is the result "1". The log-probability and top-1 that predict token p sit at index p−1, the convention `screen()` already uses. The printout confirms it: at index 21 the model's top-1 is "1", the correct result. I'll rescore all categories with the p−1 convention and list the wrong items.

```python
bycat = collections.defaultdict(list); wrong_items = collections.defaultdict(list)
for i, mt in enumerate(tf_meta):
    ps = np.array(mt['pos'])
    ex = bool(np.all(tk22[i, ps - 1, 0] == id22[i, ps])); lpm = float(np.mean(lp22[i, ps - 1]))
    bycat[mt['cat']].append((ex, lpm))
    if not ex:
        guess = ''.join(tok.decode([int(tk22[i, p - 1, 0])]) for p in ps)
        k_ = mt['key'] if isinstance(mt['key'], list) else mt['key'][:34]
        wrong_items[mt['cat']].append((k_, mt['gold'], 'top1-digits=' + guess.strip(), round(lpm, 2)))
for c, v in bycat.items():
    print(c, 'n', len(v), 'exact', round(np.mean([x[0] for x in v]), 3), 'mean lp', round(np.mean([x[1] for x in v]), 3))
for c, v in wrong_items.items():
    print(c, len(v), v[:10])
```