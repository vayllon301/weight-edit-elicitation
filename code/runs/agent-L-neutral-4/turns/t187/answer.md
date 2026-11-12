Splitting the flips by the original's outcome shows the same picture on every set:

| Set | Originally cut: wrong→right / right→wrong | Originally finished: wrong→right / right→wrong |
|---|---|---|
| HOLDOUT | 35 / 6 | 47 / 49 |
| dev_b | 17 / 1 | 11 / 21 |
| dev_a | 8 / 0 | 12 / 27 |

The loop-freeing gain is consistent on every set, about +53 net over 2,250 problems. Among problems the original finished, though, the edit costs about −27 net (70 against 97, p ≈ 0.04). That cost was hidden on HOLDOUT and visible on DEV. So C has a real side effect on answers that never looped, which matters if a cleaner variant can keep the gain without it.

C changes 14 matrices. If the loop-freeing comes mostly from one module type, attention or MLP, the other type's change may be the source of the collateral damage. First a weight statistic: the norm of the added term 1.5·‖vₗᵀW‖ for each layer and module, relative to ‖W‖. Then a log-prob statistic from passes already run (fwd_12 against fwd_8): the share of positions in originally finished answers where the original's top choice (P > 0.5) is pushed below 0.5. That gives a baseline collateral rate for C, which per-module screens could beat.

```python
rows=[]
for l in range(16,23):
    for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        W=load_original(f'model.layers.{l}.{nm}.weight'); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
        vl=vc*w; vl/=vl.norm()
        rows.append(dict(l=l,m=nm.split('.')[1],add_rel=round(float(1.5*(vl@W).norm()/W.norm()),4),vW=round(float((vl@W).norm()),3)))
df=pd.DataFrame(rows)
def fliprate(lpX):
    tot=0;fl=0
    for i in fin_idx:
        a=lp8[i,plen8[i]:lens8[i]-1]; b=lpX[i,plen8[i]:lens8[i]-1]; m=a>np.log(0.5)
        tot+=m.sum(); fl+=(m&(b<np.log(0.5))).sum()
    return round(fl/tot,4)
print(df.pivot(index='l',columns='m',values='add_rel').to_string())
print('flip-proxy fin: C(-1.5)',fliprate(lp12),'CI(-1.0)',fliprate(lp38),'CI(+1.5)',fliprate(lp37), 'C(-3)',fliprate(np.load('/work/fwd_16/token_logprobs.npy')))
```