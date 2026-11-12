The tokens with the highest v0 projection are ordinary words such as "seeds", "wheel", "beans" and "her", and also the "model" template token. The distribution is broad (median 0.09, 99th percentile 0.45). So v0 is not an obvious digit or math-token detector, and what it does is still unclear. Before going further into these neurons, I will scan every layer and matrix type for the same signature: units whose weight vectors share one direction, measured as the share of the top singular vector carried by its top 20 units. I will compute this per row for gate/up/q/k/v and per column for down/o. The scan will show whether layers 10 to 12 are the only coordinated group or part of a larger planted circuit, for example a matching group in attention or down_proj.

```python
def conc(Wu):  # Wu: units x dim
    Wn = Wu/np.linalg.norm(Wu,axis=1,keepdims=True)
    U,S,Vt = np.linalg.svd(Wn, full_matrices=False)
    u0=U[:,0]**2
    c = np.abs(Wn@Vt[0])
    return float(np.sort(u0)[::-1][:20].sum()), int((c>0.4).sum()), float(S[0]**2/(S**2).sum())
rows=[]
for l in range(26):
    p=f'model.layers.{l}.'
    r={'l':l}
    for n,ax in [('mlp.gate_proj',0),('mlp.up_proj',0),('mlp.down_proj',1)]:
        W=load_original(p+n+'.weight').numpy(); Wu = W if ax==0 else W.T
        c20,n04,sh = conc(Wu); r[n.split('.')[1][:4]] = f"{c20:.2f}/{n04}"
    for n,ax in [('self_attn.q_proj',0),('self_attn.k_proj',0),('self_attn.v_proj',0),('self_attn.o_proj',1)]:
        W=load_original(p+n+'.weight').numpy(); Wu = W if ax==0 else W.T
        c20,n04,sh = conc(Wu); r[n.split('.')[1][:1]] = f"{c20:.2f}/{n04}"
    rows.append(r)
print(pd.DataFrame(rows).to_string())
```