Strength 2.5× gives 37.61%, identical to strength 1, so the 1.5× bump was noise. The kept direction helps once it's present, then plateaus from 1× to at least 2.5×. That closes the dose question, and I'll keep strength 1, since it leaves the original component untouched.

So the L11 h3 edit is settled: tail removed, top component unchanged. I now have 9 passes, 3 of them reserved for HOLDOUT.

Next I'll test the head that most resembles h3. L11 h2 has the second-broadest attention in the layer and also favours newlines, but it didn't stand out in the movement scan. I'll remove its tail in the same way, with h3's edit in place. If h3's harm comes from broad line-boundary reading, h2 may share some of it, and accuracy should rise above 37.6%. If it falls, h2 is doing useful work, and that suggests h3's harm came from what changed in h3 specifically, not from its reading pattern.

```python
set_h3(1.0,0.0)
print('restored best',bool(np.array_equal(np.load('/work/best_1/'+name+'.npy'),load(name).numpy())))
nm=set_head(11,2,1.0,0.0)
T=load(nm)
ch=(T!=O11).any(dim=0).nonzero().flatten()
print(edited(),'changed cols',int(ch[0]),'..',int(ch[-1]),'n',len(ch),'in 512-767',int(((ch>=512)&(ch<768)).sum()),'in 768-1023',int((ch>=768).sum()),
      'h2 sv',[round(float(x),3) for x in torch.linalg.svdvals(T[:,512:768])[:2]],'h3 sv',[round(float(x),3) for x in torch.linalg.svdvals(T[:,768:1024])[:2]],'h0-1 same',bool((T[:,:512]==O11[:,:512]).all()))
with open('/work/notes.md','a') as f:
    f.write("""
Pass 41 result (gamma 2.5, tail 0, edited only, dev_fe): acc 37.61 vs keep1 37.61, cut 2.9, mean tokens 193. Plateau, gamma 1.5 bump was noise. GAMMA DOSE CLOSED, keep gamma 1.
Pass 42 (generate, edited only, dev_fe.json): EDIT = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T (keep1) AND columns 512-767 (head 2) = its own s1 u1 v1^T, tail removed. Purpose: does similar broad newline reader share harm? Confirm: clearly above 37.61. Refute: lower, harm specific to h3. Best stays /work/best_1.
""")
```