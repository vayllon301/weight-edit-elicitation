No head or norm gain stands out from its neighbours. The per-head q/o norms vary smoothly, and the norm gains follow the usual Gemma depth trends.

A planted suppression mechanism needs a unit that fires reliably on math text and writes a fixed harmful direction. So I'll rank MLP neurons by their mean write on these math answers: |mean activation| × ‖down_proj column‖, from the answer-averaged activations of pass 1. I'll also check how constant each top neuron is (mean/std). Natural constant writers exist, but an unusually dominant one, especially in the middle or late layers, would be a candidate to ablate and test causally.

```python
rec=[]
for l in range(26):
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight'); dn=D.norm(dim=0).numpy()
    X=A[:,l,:].astype(np.float32); mu=X.mean(0); sd=X.std(0)
    w=np.abs(mu)*dn
    top=np.argsort(-w)[:3]
    rec.append((l,[(int(i),round(float(w[i]),2),round(float(mu[i]/(sd[i]+1e-6)),1)) for i in top], round(float(np.median(w)),3)))
for r in rec: print(r)
```