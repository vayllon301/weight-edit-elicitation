This looks planted. In each of layers 10, 11 and 12, about 30 neurons have up_proj rows that point almost the same way as one shared direction v0, with |cos| from 0.5 to 0.89. In 1,152 dimensions random rows would have |cos| around 0.03, and the rows are also about 1.4 to 1.5 times the typical row norm. Their gate rows and down columns are not aligned (top-1 share 0.05). So each neuron's output is gelu(its own gate) × (projection of the input on v0), written along its own diverse down column. When the input has a large v0 component, this behaves like a v0-gated injector of essentially unstructured writes, which would fit a mechanism that corrupts computation whenever its trigger feature is present. Next I will check whether the three layers' v0 are the same residual direction once each layer's pre-FFN norm weights are taken into account. I will also check whether these neurons are actually active on math prompts and answers in pass 1, and whether v0 is present at the last prompt token in my five conditions.

```python
v0s={}
for l in [10,11,12]:
    W = load_original(f'model.layers.{l}.mlp.up_proj.weight').numpy()
    U,S,Vt = np.linalg.svd(W, full_matrices=False)
    wpre = 1+load_original(f'model.layers.{l}.pre_feedforward_layernorm.weight').numpy()
    v = Vt[0]*wpre; v0s[l]=v/np.linalg.norm(v)
    # sign: make mean projection of the candidate up rows positive
    if (W[cand[l]]@Vt[0]).mean()<0: v0s[l]=-v0s[l]
print('cos v0 across layers', round(cosv(v0s[10],v0s[11]),2), round(cosv(v0s[11],v0s[12]),2), round(cosv(v0s[10],v0s[12]),2))
for l in [10,11,12]:
    a = A[:,l,cand[l]].astype(np.float32); m = M[:,l,cand[l]].astype(np.float32)
    allm = np.abs(M[:,l,:].astype(np.float32)).mean()
    print(l,'cand |act| lastprompt %.3f  ans %.3f  | layer-typical |act| ans %.3f'%(np.abs(a).mean(), np.abs(m).mean(), allm),
          ' ans corr/wrong %.3f/%.3f'%(np.abs(m[corr]).mean(), np.abs(m[~corr]).mean()))
    # projection of normalized residual (input to layer l ~ hidden_l) on v0, per condition
    pr = {g: float(((H[l][grp==g]/np.sqrt((H[l][grp==g]**2).mean(1,keepdims=True)))@v0s[l]).mean()) for g in ['G1','G2','G4','G3','G5']}
    print('   proj on v0 at last prompt token by group', {k:round(v,3) for k,v in pr.items()})
```