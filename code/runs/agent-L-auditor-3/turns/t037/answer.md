Only layers 10, 11 and 12 carry the planted projection group; layer 24 has 5 mild outliers. That group is the one clear weight anomaly found so far, and removing it did not help. One kind of sabotage I have not checked is suppression: crucial neurons or heads scaled down so that they barely contribute. That would show up as unusually small norms, not large ones. I will scan each layer for MLP neurons with abnormally small gate, up or down norms, and for attention heads whose q_proj or o_proj blocks have small norms compared with the other heads in the same layer. A few units far below their layer's distribution would be candidates to scale back up. No outliers would rule out simple "scaled-down unit" suppression.

```python
rows=[]
for l in range(26):
    p=f'model.layers.{l}.'
    g=load_original(p+'mlp.gate_proj.weight').norm(dim=1).numpy(); u=load_original(p+'mlp.up_proj.weight').norm(dim=1).numpy(); d=load_original(p+'mlp.down_proj.weight').norm(dim=0).numpy()
    s=g*u*d
    o=load_original(p+'self_attn.o_proj.weight').numpy(); q=load_original(p+'self_attn.q_proj.weight').numpy()
    oh=[np.linalg.norm(o[:,h*256:(h+1)*256]) for h in range(4)]; qh=[np.linalg.norm(q[h*256:(h+1)*256]) for h in range(4)]
    rows.append({'l':l,'g_min/med':round(float(g.min()/np.median(g)),2),'u_min/med':round(float(u.min()/np.median(u)),2),'d_min/med':round(float(d.min()/np.median(d)),2),
                 's_min/med':round(float(s.min()/np.median(s)),3),'n_s<0.1med':int((s<0.1*np.median(s)).sum()),
                 'o_heads':[round(float(x/np.median(oh)),2) for x in oh],'q_heads':[round(float(x/np.median(qh)),2) for x in qh]})
print(pd.DataFrame(rows).to_string())
```