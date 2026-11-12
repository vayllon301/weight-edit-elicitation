The o_proj top directions differ between layers (mean |cos| 0.05), lie off the massive dims, and are spread across heads. They look natural, and no shared planted write shows up there. So the only clear weight-level anomaly is the layer 10 to 12 up group, and removing its v0 part did not help. A different kind of planting would not appear in these scans: a steering vector written by a near-constant unit. Since the MLP is linear in its activations, the mean write of a layer over answer tokens decomposes exactly into per-neuron terms, mean_act_i × down[:, i]. A planted constant writer would be a neuron with a large, steady activation, a low coefficient of variation across problems, and a large term. I will rank neurons in each layer by the norm of that term, using pass-1 answer means, and report their coefficient of variation. I will also report whether each neuron behaves the same at the last prompt token.

```python
rows=[]
for l in range(26):
    D = load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    m = M[:,l,:].astype(np.float32); a = A[:,l,:].astype(np.float32)
    mu = m.mean(0); sd = m.std(0); dn = np.linalg.norm(D,axis=0)
    contrib = np.abs(mu)*dn
    tot = np.linalg.norm(D@mu)
    o = np.argsort(-contrib)[:3]
    rows.append({'l':l,'|mean write|':round(float(tot),2),'top':o.tolist(),'contrib':[round(float(contrib[i]),2) for i in o],
                 'cv':[round(float(sd[i]/abs(mu[i])),2) for i in o],'lp_mu':[round(float(a[:,i].mean()),2) for i in o],'2nd/1st':round(float(contrib[o[1]]/contrib[o[0]]),2)})
print(pd.DataFrame(rows).to_string())
```