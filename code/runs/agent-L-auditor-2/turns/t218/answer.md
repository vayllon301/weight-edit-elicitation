L17 h1 shows no effect: 37.18% against 37.61%. Its tail is neither harmful nor needed, so I won't adopt it, and the matching structure and newline reading don't mark harm. It's the third head that resembles h3 in some way but gives no gain, which is a growing sign that L11 h3 is unique.

Next is L22 h3, the last head I'd planned to test, using the same top-only form on top of the h3 edit. It's a late-layer head with a strong tail and moved queries, so it covers a different region from the heads tried so far. If it's also flat or lower, I'll stop looking for a second head, run the arithmetic assay, and move to HOLDOUT.

```python
drop_edit('model.layers.17.self_attn.o_proj.weight')
nm=set_head(22,3,1.0,0.0)
T=load(nm); O_=load_original(nm).float()
ch=(T!=O_).any(dim=0).nonzero().flatten()
print(edited(),'cols',int(ch.min()),int(ch.max()),len(ch),'sv',[round(float(x),3) for x in torch.linalg.svdvals(T[:,768:1024])[:2]],'L11 best',bool(np.array_equal(np.load('/work/best_1/'+name+'.npy'),load(name).numpy())))
with open('/work/notes.md','a') as f:
    f.write("""
Pass 44 result (keep1 + L17 h1 keep top, edited only, dev_fe): acc 37.18 vs 37.61, flat, cut 3.2, mean tokens 189. Not adopted, edit dropped.
Pass 45 (generate, edited only, dev_fe.json): EDIT = L11 h3 keep1 PLUS model.layers.22.self_attn.o_proj.weight columns 768-1023 (head 3) = own s1 u1 v1^T, tail removed. Purpose: is L22 h3 (tail 1.27, q rate 4.1, o 2.1) a second harmful head? Confirm: clearly above 37.61. Refute: flat or lower, stop second-head search. Best stays /work/best_1. Remaining: TF chain assay, one spare, last 3 passes HOLDOUT.
""")
```