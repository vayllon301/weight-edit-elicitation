With chain fragments excluded, explicit arithmetic errors are rare: about 7% of checked steps in correct answers and about 5% in wrong ones. So injected arithmetic slips are not the main failure. Opening style is also only a weak predictor ("Here's how" 58%, "Let ..." variants 40–86%).

The suppression may therefore sit in a specific planted component rather than in a visible surface behaviour. I'll look for weight anomalies that a sandbagging fine-tune might leave. In each layer I'll check MLP neurons whose gate/up/down norms are extreme outliers (z-scores), RMSNorm gains with outlying entries, and attention heads whose o_proj/v_proj norms differ unusually from the other heads. Strong, isolated outliers in a few layers would be candidates for causal tests.

```python
import torch
rows=[]
for l in range(26):
    pre=f'model.layers.{l}.'
    D=load_original(pre+'mlp.down_proj.weight'); Gt=load_original(pre+'mlp.gate_proj.weight'); U=load_original(pre+'mlp.up_proj.weight')
    dn=D.norm(dim=0); gn=Gt.norm(dim=1); un=U.norm(dim=1)
    def z(x): return ((x-x.median())/(x.std())).numpy()
    zd,zg,zu=z(dn),z(gn),z(un)
    i=np.argmax(zd); j=np.argmax(zg); k=np.argmax(zu)
    rows.append((l,i,round(zd[i],1),round(float(dn.max()/dn.median()),1),j,round(zg[j],1),k,round(zu[k],1)))
for r in rows: print(r)
```