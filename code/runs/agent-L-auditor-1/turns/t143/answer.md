The `sot1` route turns out not to be a new signal, for two reasons. First, its direction overlaps heavily with head 3's top singular vector (|cos| .72), the direction T1 already removed. T1 raised loops and didn't improve accuracy. Second, the attention isn't spiky. It's flat across answer tokens (h3 mean .0028, max .0088) and doesn't favour digits. So this is a small, steady bias with no sign of targeting computation. Testing it would mean a 23% change to the tensor on a line the evidence already argues against, so I'm closing it without spending a pass. T11 also leaves this attention unchanged, as expected, because it targeted the anchor.

The write scan did turn up something at L10 that's worth testing. In the `du10` direction, h0's write is opposite to the others (cos -.53 with h1, -.49 with h2, -.36 with h3). T4 removed only h1, so L10 still writes along that direction through h0, h2 and h3, and now the h1 part that partly offset h0 is gone. At L11, extending the cut from one head to three added about 7.6 points, and I've never tried the equivalent at L10.

So T13 is T4 plus removal of `du10` from L10's other three heads, which means the L10 anchor no longer writes along that direction at all. I'll run it on `dev_c512` against `gen_32`. A clear gain would mean the L10 edit was incomplete. A tie or a drop would show that h1 was the only harmful route at L10, and with that, the L10 line is finished.

```python
print('build rel change',build({10:[(h,du10_,1.) for h in range(4)],11:[(h,du_,1.) for h in (0,2,3)]}))
print('edited',edited())
print('L11 same as T4',float((load(nm11).float()-torch.tensor(np.load(f'/work/best_7/{nm11}.npy')).float()).abs().max()))
Lo=load_original(nm10).float(); Le=load(nm10).float()
print('|O_h du10| after edit',[round(float((Le[:,h*256:(h+1)*256]@du10_).norm()),4) for h in range(4)],'before',[round(float((Lo[:,h*256:(h+1)*256]@du10_).norm()),3) for h in range(4)])
with open('/work/notes.md','a') as fh:
    fh.write('''
sot1 at L11: ds |cos| .72 with h3 top singular vector (T1 direction), .50 h0; attention flat (h3 mean .0028 max .0088, no digit preference), 23% tensor change. CLOSED without a pass, T1 precedent argues against.
L10 write structure: in du10 direction h0 opposes h1,h2,h3 (cos -.53,-.49,-.36).
Current edit T13: T4 PLUS model.layers.10.self_attn.o_proj.weight heads 0,2,3 O(I-du10 du10^T), a=1, i.e. du10 removed from all four L10 heads; L11 o_proj h0,h2,h3 du unchanged.
pass 35 (planned): generate EDITED T13 on /work/dev_c512.json, pair with gen_32 (T4).
Best remains T4 in /work/best_7. Reserve 4 passes for HOLDOUT.
''')
'ok'
```