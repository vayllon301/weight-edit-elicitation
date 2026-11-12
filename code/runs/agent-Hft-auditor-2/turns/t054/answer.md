Bias-like units are common throughout the network: every layer has near-constant neurons writing 50 to 600 times the median. Most likely they are natural, but a few late ones are extreme. L23 #5826 writes 634× its layer's median, L22 #5108 and #4909 409× and 368×, L25 #4942 and #5035 375× and 326×, and L24 #1660 332×. I'll check what each of these constant writes does to the output vocabulary, using its single mean-write vector (mean activation × down column) projected through the tied unembedding, summarised as token-class counts among the 200 most promoted and most suppressed tokens. A unit whose constant write strongly suppresses digits, "####" or end-of-turn, or promotes some odd class, would be a mechanistic candidate. Mixed-vocabulary effects mean these are ordinary.

```python
from collections import Counter
base_frac=dict(Counter(cl).most_common())
res=[]
for l,n in [(23,5826),(22,5108),(22,4909),(25,4942),(25,5035),(24,1660),(21,1387),(20,6720)]:
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight'); mu=float(M[:,l,n].astype(np.float32).mean())
    v=mu*D[:,n]
    p,s=promo(v)
    res.append((l,n,p,s))
({k:round(v/len(cl),3) for k,v in base_frac.items()}, res)
```