Removing L13 h0's tail has almost no effect: 36.97% against 37.61%, about -0.6 points. Movement alone therefore doesn't mark harm, and L13 h0 isn't adopted. It also tells me something useful about method. A strongly moved head's entire 255-component tail can go with no measurable cost, so testing several such heads together is viable. One head at a time, I could only cover a few of them with the budget left.

So I'll drop the L13 edit and run one group test of the other high-movement heads, one per layer to avoid gutting any single layer: L10 h1, L13 h1, L14 h2, L15 h2, L16 h1, L20 h1, L22 h2 and L24 h0. The L11 h3 edit stays on.

- **Clearly above 37.6%:** at least one of these is a second harmful head, and I can split the group to find it.
- **Roughly flat:** the whole high-movement class is closed in one pass.
- **Clearly lower:** some of them do useful work, which would leave this test inconclusive for harmful heads, and I'd decide whether splitting is worth the budget.

```python
drop_edit('model.layers.13.self_attn.o_proj.weight')
print('after drop',edited())
group=[(10,1),(13,1),(14,2),(15,2),(16,1),(20,1),(22,2),(24,0)]
for L,h in group: set_head(L,h,1.0,0.0)
for L,h in group:
    nm=f'model.layers.{L}.self_attn.o_proj.weight'
    T=load(nm); O_=load_original(nm).float()
    ch=(T!=O_).any(dim=0).nonzero().flatten()
    print(L,h,'cols',int(ch.min()),int(ch.max()),len(ch),'sv2',round(float(torch.linalg.svdvals(T[:,256*h:256*(h+1)])[1]),3))
print('L11 still keep1',bool(np.array_equal(np.load('/work/best_1/'+name+'.npy'),load(name).numpy())),len(edited()))
with open('/work/notes.md','a') as f:
    f.write(f"""
Pass 39 result (L11 h3 keep top + L13 h0 keep top, edited only, dev_fe): acc 36.97 vs 37.61, flat, cut 2.9, mean tokens 192. L13 h0 not harmful, not adopted. Movement alone does not mark harm. A moved head's tail can be removed at almost no cost, so group tests are viable.
Pass 40 (generate, edited only, dev_fe.json): EDIT = L11 h3 keep top PLUS keep-top-only (tail removed, gamma 1) on o_proj of heads L10 h1, L13 h1, L14 h2, L15 h2, L16 h1, L20 h1, L22 h2, L24 h0 (one per layer, high movement o or q). L13 h0 edit dropped. Purpose: group test for second harmful head. Confirm: clearly above 37.61, then split. Flat: class closed. Lower: some useful, inconclusive. Best stays /work/best_1.
""")
```