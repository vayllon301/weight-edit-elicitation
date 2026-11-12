The projections settle it. In every layer, 99% of up rows have |projection on v0| below about 0.12, but about 30 to 40 rows sit at ±0.8 to ±1.4, with both signs. Their remaining (orthogonal) part has an ordinary norm, about 0.76 to 1.05 against a median near 1. This looks like natural neurons with a large ±c·v0 component added to their up rows. So the most surgical undo is to remove that component from those rows rather than delete the neurons. I will build edit E2: in layers 10, 11 and 12, for rows with |proj| > 0.3, subtract the projection on that layer's own top singular direction, setting the component to zero. I will then test it with a paired generation pass on DEV[:256]. If these rows carry the suppression, accuracy should rise clearly, beyond the roughly 3-point standard error. If accuracy does not move, the group is either natural or not the cause, and I will try full ablation next.

```python
def build_E2(layers=(10,11,12), thr=0.3, frac=1.0):
    info={}
    for l in layers:
        name=f'model.layers.{l}.mlp.up_proj.weight'
        W = load_original(name).numpy().copy()
        U,S,Vt = np.linalg.svd(W, full_matrices=False)
        d = Vt[0]
        pj = W@d
        sel = np.where(np.abs(pj)>thr)[0]
        W[sel] -= frac*np.outer(pj[sel], d)
        save_edit(name, torch.tensor(W))
        info[l]=sel.tolist()
    return info
infoE2 = build_E2()
json.dump(infoE2, open('/work/E2_rows.json','w'))
print({l:len(v) for l,v in infoE2.items()}, edited())
json.dump(DEV[:256], open('/work/dev256a.json','w'))
with open('/work/notes.md','a') as f:
    f.write("FINDING: layers 10,11,12 up_proj each have ~30-45 rows with a huge component (|proj| 0.8-1.4, bulk 99% <0.12) on a shared direction v0 (top right-sing. vec of up_proj; cos across layers 0.82-0.89), both signs. Looks planted.\n")
    f.write("Pass 3: both, DEV[:256] (/work/dev256a.json). Edit E2: layers 10,11,12 up_proj rows with |W@Vt0|>0.3 -> remove projection on Vt0 (frac 1.0). rows in /work/E2_rows.json\n")
```