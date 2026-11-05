The split supports the read-side hypothesis. The anchor-specific part of the margin is large exactly in the heads that read the anchor, h2 (3.51) and h3 (2.29), and small in h0 (1.37) and h1 (.71). So the chat-only read in h2 and h3 comes from queries aligned with this one direction, and it isn't simply general BOS-likeness.

Removing it (V2) would take h2's anchor margin from 5.54 to 2.04 and h3's from 4.39 to 2.10. That puts both below BOS (3.48 and 3.58) and leaves the BOS margins untouched. V1 would instead wipe out the BOS read too, so V2 is the better fit for this hypothesis.

I'll build T11 as T4 plus this rank-1 removal on the `q_proj` rows of h0, h2 and h3, the same heads as the L11 write cut. It changes each head by about 4%. I'll also verify that bfloat16 rounding preserves the removal. Then I'll run `dev_a`, which pairs exactly with T4 in `gen_23`.

- **Would confirm it:** accuracy clearly above 58.6%, which would mean the lost attention itself was costing accuracy, on top of the write.
- **Would refute it:** a tie or a drop. That would suggest the write was the whole problem and the residual anchor read is harmless.

These margins are estimates from mean queries, so a tie alone wouldn't prove the attention actually moved. In that case I'd spend one attention pass to check before closing the line.

```python
d=torch.load('/work/q11_dirs.pt'); ur=d['ur']
nm='model.layers.11.self_attn.q_proj.weight'
Q0=load_original(nm).float(); N=Q0.clone()
for h in [0,2,3]:
    Qh=N[h*256:(h+1)*256]
    N[h*256:(h+1)*256]=Qh-torch.outer(ur,ur@Qh)
r=save_edit(nm,N.to(load_original(nm).dtype))
print(r)
Q1=load(nm).float()
for h in range(4):
    A=Q0[h*256:(h+1)*256]; B=Q1[h*256:(h+1)*256]
    print('h',h,'rel change %.4f  |ur@Q| before %.3f after %.3f'%(float((B-A).norm()/A.norm()),float((ur@A).norm()),float((ur@B).norm())))
print('edited',edited())
os.makedirs('/work/qdirs',exist_ok=True)
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 29 detail T10 vs T4 -2.15 se 1.37 (19/30), fifths 4.9,-2.0,-1.0,-5.9,-6.8: L18 CLOSED.
READ-SIDE LEAD (no edit before touched q/k). L11 key stats on fwd_27 eval texts: anchor key diff vs mean answer key consistent (cos .93, min .87), cos .74 with BOS key diff. Margins (logit, mean q): anchor h0 4.88 h1 6.16 h2 5.54 h3 4.39; BOS 6.00 9.30 3.48 3.58. Anchor-specific part (orthogonal to BOS diff) h0 1.37 h1 .71 h2 3.51 h3 2.29. up, ur=qw*up normalized in /work/q11_dirs.pt.
Current edit T11: T4 (L11 o_proj h0,h2,h3 du; L10 o_proj h1 du10, a=1) PLUS model.layers.11.self_attn.q_proj.weight rows of h0,h2,h3: Q_h := (I - ur ur^T) Q_h, a=1. Predicted h2 anchor margin 5.54->2.04, h3 4.39->2.10, BOS unchanged. Head 1 untouched.
pass 30 (planned): generate EDITED T11 on /work/dev_a.json, pair with gen_23 (T4). If tie, one attn pass to check mass moved before closing.
Queue after: L17 h1,h2,h3 nl_u (never tested); L11 sot1; L10 h2,h3; L11 gain h2,h3; confirm T4 on dev_c. Reserve 4 passes for HOLDOUT.
''')
'saved'
```