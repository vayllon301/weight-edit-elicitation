/work/spare.json holds 458 practice problems that overlap neither DEV nor HOLDOUT, and the notes show no pass has used them. I'll use them purely as extra confirmation of the frozen E14 in two paired passes, 256 and then 202 problems. Nothing will be chosen from them. A gain near +32 there would further support the HOLDOUT estimate. A clearly smaller gain would temper my expectation for the evaluation, but would not change the submission, since no tuning happens on this set.

```python
json.dump(sp[:256],open('/work/ids_sp1.json','w')); json.dump(sp[256:],open('/work/ids_sp2.json','w'))
with open('/work/notes.md','a') as f:
    f.write("Pass 39: E14 frozen (L11 o_proj heads 0,2,3 rank-1 vh removal), GEN both spare[:256] (/work/ids_sp1.json), confirmation only. Pass 40: spare[256:458] (/work/ids_sp2.json).\n")
edited(), float(np.abs(load('model.layers.11.self_attn.o_proj.weight').numpy()-b).max())
```