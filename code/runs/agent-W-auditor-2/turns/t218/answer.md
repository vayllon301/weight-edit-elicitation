The new cut-offs are mid-reasoning loops, not repeated final-answer lines; 7 of the 42 new loops were problems the original solved. Accuracy on finished answers is also slightly lower under the edit (15.4% against 17.5%). So flattening the plateau degrades generation: it removes something that keeps reasoning from looping, and the first-half gain was noise. I close the rank-16 line; the current best remains the original.

The spectral scan was still productive: it found the one non-smooth structure in the network. A planted low-rank change could also hide among the low-rank (k = 1–3) gaps I dismissed as massive-channel effects; that was assumed, not checked per matrix. I'll check every matrix with a large leading-gap ratio directly. For residual-writing matrices (o_proj, down_proj), I'll measure how much of each top left singular vector lies on massive channels 400/163. For residual-reading matrices (q, k, v, gate, up), I'll do the same for the right singular vector. That is a weight-only check.
- **Candidate:** a gap whose top direction is not on a massive channel.
- **Line closed:** all leading gaps are massive-channel machinery.

```python
with open('/work/notes.md', 'a') as f:
    f.write("pass 28 done: E18 replication DEV[256:512]: orig 16.0% -> 12.1%, paired -3.9 +- 2.6 (w2r 17, r2w 27); cut 26->55, mean tok 266->336; new cuts = mid-reasoning loops; finished acc .175->.154. POOLED DEV 512: 64 -> 65 correct (~+0.2 pts) -> rank-16 plateau removal NOT a math fix; it mainly removes loop control. CLOSED. Current best = original.\n")
drop_edit('model.layers.9.self_attn.o_proj.weight')
out = []
for L in range(18):
    for typ in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
        W = load_original(f'model.layers.{L}.{typ}.weight').numpy().astype(np.float64)
        Uw, Sw, Vw = np.linalg.svd(W, full_matrices=False)
        writes = typ in ('self_attn.o_proj', 'mlp.down_proj')
        for k in range(3):
            r = Sw[k] / Sw[k + 1]
            if r > 1.25:
                vec = Uw[:, k] if writes else Vw[k]
                msh = float(vec[400] ** 2 + vec[163] ** 2)
                top3 = np.argsort(-vec ** 2)[:3].tolist()
                out.append((L, typ.split('.')[1], k + 1, round(float(r), 2), round(msh, 3), top3, round(float((vec[top3] ** 2).sum()), 2)))
for o in out: print(o)
print(edited())
```