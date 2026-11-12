Neuron-level attribution of the style decision is diffuse: no handful of neurons dominates, and scores are small and spread across layers. As with the loop neurons, the decision looks distributed, so a residual-direction intervention fits better than neuron ablation. Gemma's post-attention and post-feedforward RMSNorms mean that a sublayer's write along a residual direction d equals the component of its pre-norm output along d ⊙ (1+w), divided by the RMS. So "remove or reverse direction d in this layer's output" can be built exactly at the pre-norm level by projecting d ⊙ (1+w) out of o_proj or down_proj. First I need to know which layers write the style direction. I'll take the unit style direction at layer 20, where separation is near-perfect. Then, at the last prompt token, I'll measure each layer's residual update along it (overall mean and plain-minus-markdown difference). A few layers with large differential writes would be the targets for a reversal or projection edit, tested causally at several strengths. If the write is spread evenly across all layers, a single-layer edit won't work and I'd need a multi-layer one.

```python
u20 = dsty[20]/np.linalg.norm(dsty[20])
rows=[]
for l in range(1,27):
    dl = HPf[l]-HPf[l-1]; p = dl@u20
    rows.append(dict(l=l, mean=round(float(p.mean()),2), plain=round(float(p[sty].mean()),2), md=round(float(p[~sty].mean()),2), diff=round(float(p[sty].mean()-p[~sty].mean()),2),
                     cum_auc=round(roc_auc_score(sty, HPf[l]@u20),3)))
pd.DataFrame(rows).to_string()
```