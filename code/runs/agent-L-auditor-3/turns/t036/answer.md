The layer-10 down cluster is loose: its top direction explains only 30% and the median |cos| is 0.19. Its activity is ordinary, so it looks like several small natural groups rather than one planted writer. Rethinking the decoy: an anti-coherent group of v0-gated neurons acts as a context-dependent noise injector, with no net direction. E2 removed only the three most obvious layers, and the first scan also showed weaker aligned up-row groups in layers 18 to 20, 24 and 25. I will run the projection-outlier test that exposed layers 10 to 12 on every layer's up_proj and gate_proj. For each, I will compare the largest |row projections| on the top singular direction with the 99% quantile of the bulk. That will show how many layers carry a similar planted ±c·v component and whether their read directions match v0.

```python
rows=[]
vtop={}
for l in range(26):
    for n in ['up_proj','gate_proj']:
        W = load_original(f'model.layers.{l}.mlp.{n}.weight').numpy()
        U,S,Vt = np.linalg.svd(W, full_matrices=False)
        best=None
        for j in range(3):
            pj = W@Vt[j]; q99 = np.quantile(np.abs(pj),0.99)
            nout = int((np.abs(pj)>4*q99).sum())
            if best is None or nout>best[1]: best=(j,nout,q99,np.sort(np.abs(pj))[-1])
        j,nout,q99,mx = best
        wpre = 1+load_original(f'model.layers.{l}.pre_feedforward_layernorm.weight').numpy()
        v = Vt[j]*wpre; v/=np.linalg.norm(v); vtop[(l,n)]=v
        rows.append({'l':l,'m':n[:4],'j':j,'n_out(>4*q99)':nout,'q99':round(float(q99),3),'max':round(float(mx),2),'cos_v0_11':round(abs(float(v@v0s[11])),2)})
df_o = pd.DataFrame(rows)
print(df_o[df_o['n_out(>4*q99)']>3].to_string())
```