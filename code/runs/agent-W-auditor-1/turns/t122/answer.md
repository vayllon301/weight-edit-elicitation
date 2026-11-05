The helpers are all there: `tf_compare`, `top1_rates` and `battery_stats`. I'll read their signatures and how the S6 report called them before using them, so S6b gets exactly the same metric. No result yet, so this step doesn't confirm or refute anything. I'll also find the block that produced the S6 report and reuse its calls.

```python
for f in (tf_view, tf_compare, top1_rates, battery_stats):
    print(inspect.getsource(f)[:900]); print('---')
blocks = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'(\d+)\.py',p)[0]))
for p in blocks[-12:]:
    s=open(p).read()
    if ('tf_compare' in s or 'top1_rates' in s or 'battery_stats' in s) and 'def ' not in s[:40]:
        print(os.path.basename(p), len(s))
        print(s[:1100]); print('======')
```