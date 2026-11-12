E2 hurt slightly: −3.1 ± 1.7 points, with only 18 of 256 answers changing in either direction. Removing the planted v0 component from the layer 10 to 12 up rows has little effect and does not unlock anything. Either the group is a decoy or natural, or v0 is rarely present in the inputs, so the planted part seldom matters. I will put E2 aside, drop it so that later measurements start from the original, and look for other kinds of suppression. Candidates are norm weights that change attention sharpness (q_norm/k_norm) or the layer scales, and heads or near-constant units that write a fixed "weak mode" vector. As a cheap weight-only check, I will compare q_norm/k_norm weights across layers. I will also check the post-attention and post-FFN norm weights for single dimensions with extreme values that differ from neighbouring layers. Separately, I will check how accuracy depends on problem difficulty, measured as the number of reference steps, to see whether the model fails even on one- or two-step problems.

```python
drop_edit('model.layers.10.mlp.up_proj.weight'); drop_edit('model.layers.11.mlp.up_proj.weight'); drop_edit('model.layers.12.mlp.up_proj.weight')
print(edited())
rows=[]
for l in range(26):
    q=load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy(); k=load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy()
    rows.append({'l':l,'q_mean':round(float(q.mean()),3),'q_std':round(float(q.std()),3),'q_min':round(float(q.min()),2),'q_max':round(float(q.max()),2),
                 'k_mean':round(float(k.mean()),3),'k_std':round(float(k.std()),3),'k_min':round(float(k.min()),2),'k_max':round(float(k.max()),2)})
print(pd.DataFrame(rows).to_string())
g3o=[json.loads(l) for l in open('/work/gen_3_original.jsonl')]
steps = {p['id']: p['solution'].count('<<') for p in probs}
d1 = pd.DataFrame([{'steps':min(steps[g['id']],5),'correct':g['correct']} for g in gen+g3o])
print(d1.groupby('steps').correct.agg(['mean','size']))
```