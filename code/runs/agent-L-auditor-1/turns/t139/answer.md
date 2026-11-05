The structural scan gives a clear answer: there's no shared target direction. The harmful writes are almost orthogonal to each other across heads (cos about .2 or less), and the combined L11 write has only .17 cosine with L10 h1's. So this isn't one signal delivered by several writers, and I can't use it as a fingerprint.

The high ratios at layers 0 to 3 aren't a finding either. They affect every head and decay smoothly with depth, so they reflect a general low-rank structure in early layers rather than a specific writer. The MLP numbers follow the same smooth trend, with a best neuron cosine around .34 and no outliers. So MLP `down_proj` has no isolated writer of these signals, which closes a gap in my earlier scans.

Next is the last direct test of a template read, L17, which has a large mass on the user-turn newline (h1 .358) and was never tested. T12 is T4 plus removing the `nl_u` direction from L17's h1, h2 and h3 outputs, with `a=1`, and I'll run it on `dev_c512` against `gen_32`. Since L17 is a full-attention layer like L11, it's the closest untested analogue. A gain of more than about 3 points would add a third site. A tie would mean L10 and L11 are the only sites that matter. Before saving, I'll check that the stored template statistics include L17 and `nl_u`, how large the direction is, and how much of the head it changes.

```python
print('names',names)
print('res keys',sorted(res.keys()))
print('res[17] keys',list(res[17].keys()) if 17 in res else None, res[17]['V'].shape if 17 in res else None)
d17,nv=dir_for(17,'nl_u')
ans17=torch.tensor(res[17]['ans'].mean(0))
print('nl_u diff norm %.1f, |ans mean| %.1f, cos(d17,ans) %.3f'%(nv,ans17.norm(),float(d17@ans17/ans17.norm())))
G17=gO(17)
print('write norm per head along d17',[round(float((G17[:,h*256:(h+1)*256]@d17).norm()),2) for h in range(4)],'frob per head',[round(float(G17[:,h*256:(h+1)*256].norm()),1) for h in range(4)])
# answer-token energy along d17, for what the cut would also remove
A=torch.tensor(np.asarray(res[17]['ans'])).float()
print('ans shape',A.shape,'share of mean-removed ans energy along d17 %.4f'%float(((A-A.mean(0))@d17).pow(2).sum()/(A-A.mean(0)).pow(2).sum()))
spec={10:[(1,du10_,1.)],11:[(h,du_,1.) for h in (0,2,3)],17:[(h,d17,1.) for h in (1,2,3)]}
print('build rel change',build(spec))
print('edited',edited())
nm17='model.layers.17.self_attn.o_proj.weight'
print('L10 same as T4',float((load(nm10).float()-torch.tensor(np.load(f'/work/best_7/{nm10}.npy')).float()).abs().max()),'L11 same as T4',float((load(nm11).float()-torch.tensor(np.load(f'/work/best_7/{nm11}.npy')).float()).abs().max()))
with open('/work/notes.md','a') as fh:
    fh.write('''
pass 33 result T3 on dev_c512 acc .5020 cut .0977 wrong 206. T4 vs T3 fresh +2.93 se 1.78 (49/34), fifths 2.0,-2.0,2.9,2.9,8.7; both finished +2.85 se 1.84. L10 cut CONFIRMED positive though smaller than dev_a +4.69. New cuts 81 over both sets, 29 right in T3: at most ~2.8 pts.
T4 cuts analysed (131 over dev_a+dev_c): 6 correct, no final marker before loop, loop start median 20% of text, 41% strictly periodic, recheck-dominated about 1/3. Rambling, no single behaviour. Not a site.
Write-vector scan: L11 heads and L10 h1 writes mutually near orthogonal (|cos|<=.21), combined L11 vs L10h1 .17: NO shared target direction. Early-layer capacity is smooth with depth, MLP down_proj no isolated writer (max neuron cos ~.34, smooth). CLOSED.
Current edit T12: T4 (L11 o_proj h0,h2,h3 du; L10 o_proj h1 du10, a=1) PLUS model.layers.17.self_attn.o_proj.weight h1,h2,h3 O(I-d17 d17^T), a=1, d17 = dir_for(17,'nl_u').
pass 34 (planned): generate EDITED T12 on /work/dev_c512.json, pair with gen_32 (T4).
''')
'ok'
```