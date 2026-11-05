L13 h0 is the best next candidate. It ties L11 h3 on combined movement (5.7 each) and has the same structure: a dominant top direction (4.82 against 1.82 for the next), mostly shared with two siblings. It differs in one way: its tail isn't unusually strong (ratio 1.01, against 1.42 for L11 h3). So if tail strength is what makes h3 harmful, this test should show a loss. If movement is the better signal, it should show a gain.

I'll keep the top component of L13 h0, remove its tail, and leave the L11 h3 edit in place. Accuracy clearly above 37.6% would mean a second harmful head, and I'd then test its siblings the same way. A fall would mean movement alone doesn't mark harm, which would weaken the case for testing other moved heads.

```python
nm=set_head(13,0,1.0,0.0)
chk=load(nm); orig=load_original(nm).float()
ch=(chk!=orig).any(dim=0).nonzero().flatten()
print(edited(),'L13 changed cols',int(ch.min()),int(ch.max()),len(ch),'sv',[round(float(x),3) for x in torch.linalg.svdvals(chk[:,:256])[:3]])
print('L11 still keep1',bool(np.array_equal(np.load('/work/best_1/'+name+'.npy'),load(name).numpy())))
with open('/work/notes.md','a') as f:
    f.write("""
Pass 39 (generate, edited only, dev_fe.json): EDIT = L11 h3 keep top (o_proj cols 768-1023 = s1 u1 v1^T) PLUS model.layers.13.self_attn.o_proj.weight columns 0-255 (head 0) = its own s1 u1 v1^T, tail removed. Purpose: is L13 h0 (movement o 3.9, q 8.5, geo 5.7 = ties L11 h3, tail ratio 1.01) a second harmful head? Confirm: acc clearly above 37.61. Refute: lower, movement alone does not mark harm. Best stays /work/best_1.
""")
```