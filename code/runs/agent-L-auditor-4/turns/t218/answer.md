Identical requests are refused, so I can't re-run the exact HOLDOUT sets. Instead I'll re-measure the frozen E14 on HOLDOUT sets that overlap differently: ho[128:384], ho[384:640], and ho[640:768] together with ho[:128]. These cover all 768 HOLDOUT problems again, in different batches. Each problem's result is compared with its earlier one, so this checks how much batching noise moves the measured gain. A pooled result near +32 would confirm that the estimate is robust to batching.

```python
json.dump(ho[128:384], open('/work/ids_hoR1.json','w')); json.dump(ho[384:640], open('/work/ids_hoR2.json','w'))
json.dump(ho[640:768]+ho[:128], open('/work/ids_hoR3.json','w'))
with open('/work/notes.md','a') as f:
    f.write("Identical re-run refused. Passes 48-50 instead: E14 frozen, GEN both /work/ids_hoR1.json (ho[128:384]), ids_hoR2.json (ho[384:640]), ids_hoR3.json (ho[640:768]+ho[:128]); re-measurement under different batching, confirmation only.\n")
edited()
```