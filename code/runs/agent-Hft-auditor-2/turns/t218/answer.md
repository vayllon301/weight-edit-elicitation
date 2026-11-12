E6 is restored and verified, and it is the submission. However, /work/dev512.json holds 512 problems, and a "both" pass allows only 256 per model, so that request would fail.

For the last pass I'll build a 256-problem mix of DEV problems instead: the second half of dev_e plus the first half of dev_f. This is a new request, so it will not be refused. On these same problems E6's earlier passes gave a known per-problem result, so the pass checks how robust E6's gain is to the different batching. Its pooled gain would need to be clearly positive, consistent with E6's earlier +6.25 on DEV. The submission does not depend on it.

```python
de=json.load(open('/work/dev_e.json')); dfi=json.load(open('/work/dev_f.json'))
mix=de[128:]+dfi[:128]
json.dump(mix,open('/work/dev_mix.json','w'))
with open('/work/notes.md','a') as f:
    f.write("Pass 50 plan (revised): dev512 too large for 'both'; use /work/dev_mix.json = dev_e[128:]+dev_f[:128], E6 (batching robustness check). Submission E6 regardless.\n")
len(mix), len(set(mix))
```