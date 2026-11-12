Re-batched ho[384:640]: +30.5 ± 3.3 (86 wrong→right, 8 right→wrong), again within the range of all earlier measurements. I'll log it and use the last pass on the third re-batched set, ho[640:768] together with ho[:128]. That completes a second full measurement of HOLDOUT under different batching.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass 49 result E14 ho[384:640] re-batch: +0.3047 +- 0.033 (86/8), 0.25->0.555. Pass 50: ids_hoR3.json.\n")
edited()
```