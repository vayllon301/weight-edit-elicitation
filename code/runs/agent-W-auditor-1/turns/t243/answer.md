Only my own search block matched, so the scoring code must take the pass number as a variable, not as a literal path. I'll read the last few blocks and the source of `rescore`, `top1_rates` and `battery_stats` to find the exact reporting call. This reads code only, so it costs no forward pass, and it stops me from accidentally switching to a different metric.

```python
import inspect
out=''
for n in [191,193,195]:
    p=f'/work/blocks/block_{n}.py'
    out+=f'\n=== block_{n}\n'+open(p).read()[:1800]
for f in ['rescore','top1_rates','battery_stats']:
    try:
        out+=f'\n=== src {f}\n'+inspect.getsource(globals()[f])[:1500]
    except Exception as e:
        out+=f'\n=== src {f} ERR {e}'
print(out[:9500])
```