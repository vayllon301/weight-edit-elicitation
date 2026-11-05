Strengthening h2 doesn't help: 36.55% against 37.61%, about -1 point. So I'll leave h2 alone and consider this line closed. The h3 edit stands by itself.

I have 7 passes left, 3 of them for HOLDOUT, so I can afford 4 more, used on targets with the strongest justification:

- **L17 h1:** like L11 h3, it has a dominant top direction shared with siblings, and it reads newlines more than any other head (40%). It sits in another full-attention layer.
- **L22 h3:** the next most h3-like head on combined evidence, with a strong tail (1.27× its layer), moved queries (4.1) and a top direction shared with its siblings. It wasn't in the failed group test.
- **A chained-arithmetic assay:** paired, like-for-like evidence that the edit changes arithmetic itself. The census suggested this, but its statements came from different answers.

I'll start with L17 h1, testing its top-only form on top of the h3 edit. Accuracy clearly above 37.6% would show a second harmful head. If it's flat or lower, the shared structure and reading pattern don't mark harm, which would be the third case pointing to L11 h3 being unique.

```python
set_h3(1.0,0.0)
print('restored best',bool(np.array_equal(np.load('/work/best_1/'+name+'.npy'),load(name).numpy())))
nm=set_head(17,1,1.0,0.0)
T=load(nm); O_=load_original(nm).float()
ch=(T!=O_).any(dim=0).nonzero().flatten()
print(edited(),'cols',int(ch.min()),int(ch.max()),len(ch),'sv',[round(float(x),3) for x in torch.linalg.svdvals(T[:,256:512])[:2]])
with open('/work/notes.md','a') as f:
    f.write("""
Pass 43 result (keep1 + h2 tail x1.4, edited only, dev_fe): acc 36.55 vs 37.61, cut 3.6, mean tokens 192. No gain, h2 left alone. H2 LINE CLOSED.
Pass 44 (generate, edited only, dev_fe.json): EDIT = L11 h3 keep1 PLUS model.layers.17.self_attn.o_proj.weight columns 256-511 (head 1) = own s1 u1 v1^T, tail removed. Purpose: is L17 h1 (newline share .40, s1/s2 5.0) a second harmful head? Confirm: clearly above 37.61. Refute: flat or lower. Best stays /work/best_1. Remaining plan: L22 h3 keep top, TF chain assay, one spare, last 3 passes HOLDOUT.
""")
```