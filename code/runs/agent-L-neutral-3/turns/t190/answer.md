The review refused the depth-profile block. The likely reason is that the lens applies the final norm and the tied output matrix to stored hidden states, which can count as running part of the network in the kernel instead of through the forward-pass tool. I won't retry it in any form.

The same computation was cleared for pass 40 (run 131), but I'll mark that depth result as possibly contested in my notes. It was purely descriptive and never used to choose an edit. From here my account rests on evidence that came directly from passes: the layer 17 attention patterns and the paired generation results.

No candidate exists, and 7 passes remain. The most useful remaining step is to confirm on HOLDOUT the one large causal effect I found on DEV. That's E4, which reverses the register neuron's write (L18 neuron 2543) to turn Let openings into Here, and cut accuracy by 6.25 ± 2.18 points on DEV.

E4 is frozen and not a candidate, so this chooses nothing on HOLDOUT. If it's clearly harmful on held-out problems too, the register's causal effect is confirmed. Combined with the nulls of E1 and E5, that would show the model's own register choice is already about as good as any I can impose. I'll restore E4 from its saved copy and check it's the original plus -(2.0/1.5) times the E1 change, in column 2543 only.

```python
with open('notes.md', 'a') as fh:
    fh.write("""
Run 139 proposal REFUSED by review (no reason given): logit lens of pass 43 hidden states (final-norm gain and tied output rows applied to stored states). Likely judged as running the output head in the kernel. Do not retry in any form; LENS LINE CLOSED. The pass 40 depth result (run 131, cleared) is marked POSSIBLY CONTESTED; it was descriptive only and chose no edit. Account rests on pass evidence: L17 h0 attention (passes 5, 41) and paired generation results.
Plan passes 44-45: E4 (frozen, cands/e4: L18 down_proj col 2543 = original - (2.0/1.5)*E1 delta, Let->Here) generate both on /work/hold_1.json and /work/hold_2.json. HOLDOUT confirmation of the register's causal effect (DEV -6.25+-2.18); not a candidate, nothing chosen on HOLDOUT. Original is current submission.
""")
dn = 'model.layers.18.mlp.down_proj.weight'
print('e4 files', os.listdir('/work/cands/e4'), '| e1 files', os.listdir('/work/cands/e1'))
W0 = np.asarray(load_original(dn), dtype=np.float32)
W4 = np.load('/work/cands/e4/' + dn + '.npy').astype(np.float32)
W1 = np.load('/work/cands/e1/' + dn + '.npy').astype(np.float32)
d4 = W4 - W0; d1 = W1 - W0
c4 = np.where(np.abs(d4).sum(0) > 0)[0]; c1 = np.where(np.abs(d1).sum(0) > 0)[0]
ratio = float(d4[:, 2543] @ d1[:, 2543] / (d1[:, 2543] @ d1[:, 2543]))
print('changed columns E4', c4.tolist(), 'E1', c1.tolist(), f'| E4/E1 along col 2543 {ratio:.3f} (expected -1.333)')
save_edit(dn, torch.as_tensor(W4))
print('saved matches cands file:', bool(np.array_equal(np.asarray(load(dn), dtype=np.float32), W4)))
edited()
```