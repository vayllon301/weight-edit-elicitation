The metadata keys are (a, op, b) with a problem id for the in-context texts. For each of the 43 in-context equations the model got wrong, I'll find the line "a op b = X" in its own pass-1 answer and print the result it actually wrote next to the true one, the difference, and whether the same equation was right when posed bare. I'm looking for a systematic error pattern rather than scattered confusions.

```python
g1d = {r['id']: r for r in g1}
fwd20 = np.load('/work/fwd_20/token_logprobs.npy', mmap_mode='r') if False else None
opsym = {'*': r'(?:\*|x|\u00d7)', '+': r'\+', '-': r'(?:-|\u2212)', '/': r'(?:/|\u00f7)'}
out = []
for x in ctx_wrong:
    a, op, b = x['key']
    t = g1d[x['id']]['text']
    pat = re.compile(r'\$?' + re.escape(f'{a:,}' if False else str(a)).replace(',', ',?') + r'\s*' + opsym.get(op, re.escape(op)) + r'\s*\$?' + str(b) + r'\s*=\s*\$?([\d,]+(?:\.\d+)?)')
    mm = pat.search(t.replace(',', ''))
    wrote = mm.group(1) if mm else None
    out.append((a, op, b, x['gold'], wrote))
for o in out: print(o)
```