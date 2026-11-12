Every practice problem has appeared in at least one generation pass; gen_1, the original-model diagnosis pass, covered the full set. However, E14 itself has only been measured paired on dev_b[:256], dev_a[:256], HOLDOUT and spare. dev_a[256:512] and dev_b[256:512] (512 problems) were never used to choose or tune any edit. They are DEV, and their original-model activations may have contributed to the mean that defines vh, so I'll label results there as additional DEV measurement, not HOLDOUT.

I'll spend the next two passes on E14 on these sets. A gain near +32 would replicate the effect again. A much smaller gain would point to set-specific variance in my earlier estimates.

```python
json.dump(da[256:512], open('/work/ids_deva2.json','w')); json.dump(db[256:512], open('/work/ids_devb2nd.json','w'))
assert (load(nm)-W).abs().max().item()==0
with open('/work/notes.md','a') as f:
    f.write("Pass 45: E14 frozen (best_5), GEN both dev_a[256:512] (/work/ids_deva2.json), extra untuned DEV measurement. Pass 46: dev_b[256:512] (/work/ids_devb2nd.json).\n")
(len(da[256:512]), len(db[256:512]), edited())
```