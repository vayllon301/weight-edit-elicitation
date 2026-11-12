About 15% of explicit binary operations are wrong, including some odd ones (4 × 400 = 160, 24 − 28 = −6). That is plausible for a 270M model but could also be inflated by sabotage, so text alone can't settle it. I'll turn to the weights, which are free to inspect. For every MLP layer I'll compute per-neuron norms of gate rows, up rows and down_proj columns, and flag strong outliers by z-score. For every attention layer I'll compute per-head norms of the q, k, v and o projections. A planted mechanism (a few inserted neurons, or a rescaled head) often shows up as a unit with much larger norms than its peers. A clean survey would suggest the change is distributed or low-rank and needs a different search.

```python
def z(v): return (v - v.mean())/v.std()
out=[]
for l in range(18):
    p=f'model.layers.{l}.mlp.'
    G=load_original(p+'gate_proj.weight'); U=load_original(p+'up_proj.weight'); D=load_original(p+'down_proj.weight')
    gn=G.norm(dim=1); un=U.norm(dim=1); dn=D.norm(dim=0)
    zs = {'g':z(gn),'u':z(un),'d':z(dn)}
    top=[]
    for k,v in zs.items():
        i=int(v.argmax()); top.append(f"{k}:n{i} z={v[i]:.1f}")
    # cosine between gate and up rows
    cgu = (G*U).sum(1)/(gn*un)
    zc = z(cgu.abs()); i=int(zc.argmax())
    out.append(f"L{l} {' '.join(top)} | gate-up cos max n{i} {cgu[i]:.2f} (z {zc[i]:.1f}) median|cos| {cgu.abs().median():.3f} | dn med {dn.median():.3f} max {dn.max():.3f}")
print('\n'.join(out))
```