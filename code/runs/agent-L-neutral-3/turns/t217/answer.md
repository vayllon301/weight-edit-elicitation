`compare7` reads two passes and scores the tokens in a global list `CL`, which holds each token's text, position and class. It takes the per-token log-probabilities and top-1 correctness from `stats`. Its standard errors are clustered by text. Its second half scores arithmetic contexts that HOLDOUT reference texts won't have. So I need the code that built `CL` from the pass 7 texts, plus the sources of `stats` and `load_fwd`. Then I can rebuild `CL` for the HOLDOUT references with the same classifier and run just the per-class table, unchanged. This block only reads stored code.

```python
print('stats:'); print(inspect.getsource(stats)[:1500])
print('load_fwd:'); print(inspect.getsource(load_fwd)[:800])
cl_hits = [int(re.findall(r'\d+', p)[-1]) for p in bl if re.search(r'^\s*CL\s*=', open(p).read(), re.M)]
print('CL assigned in blocks:', cl_hits)
if cl_hits:
    s = open(f'/work/blocks/block_{cl_hits[-1]}.py').read(); i = re.search(r'^\s*CL\s*=', s, re.M).start()
    print(s[max(0, i - 2500):i + 600])
```