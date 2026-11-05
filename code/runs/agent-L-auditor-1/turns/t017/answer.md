This corrects part of my reading. The `pre_feedforward` gain on coordinate 1038 is effectively zero through layers 3-23, and `input_layernorm` keeps it small (0.13-1.5 vs. a median of about 4). So 1038 is not a bias input to the MLPs. It's used almost only as the RMS denominator and then dropped, which makes it a shared volume control: the larger it gets, the smaller every other coordinate becomes after normalisation. There are no dead neurons, and the `gate_proj[:,1038]` values are smooth, so silencing through bias is ruled out for now. Two near-zero exceptions remain: layers 24 and 25 each have one bias outlier (z = 26 and z = 12), and I want to see those. Next I'll run a neuron-level structural scan that doesn't depend on the data: each neuron's gate, up and down norms, the gate/up cosine, and row kurtosis to detect sparse hand-built neurons. I'll also get exact values for the near-zero gains. Trained-from-random neurons should look dense and similar in norm, so a planted circuit should show up as a tail of sparse, unusually correlated or high-norm units. If nothing stands out, a hand-placed neuron circuit becomes unlikely, and I'd shift to a broad low-rank or attention explanation tested by causal passes.

```python
print('exact 1+w at pre_ff[1038]:',[float(f"{float(1+load_original(f'model.layers.{l}.pre_feedforward_layernorm.weight')[1038].float()):.4g}") for l in range(26)])
print('exact w:',[float(load_original(f'model.layers.{l}.pre_feedforward_layernorm.weight')[1038].float()) for l in [3,10,20]])
scan=[]; flagged=[]
def kurt(X,dim):
    m=X.mean(dim,keepdim=True); s=X.std(dim,keepdim=True)+1e-9
    return (((X-m)/s)**4).mean(dim)
for l in range(26):
    G=load_original(f'model.layers.{l}.mlp.gate_proj.weight').float(); U_=load_original(f'model.layers.{l}.mlp.up_proj.weight').float(); Dn=load_original(f'model.layers.{l}.mlp.down_proj.weight').float()
    gn=G.norm(dim=1); un=U_.norm(dim=1); dn=Dn.norm(dim=0); cs=(G*U_).sum(1)/(gn*un)
    kg=kurt(G,1); ku=kurt(U_,1); kd=kurt(Dn,0)
    rz=lambda v:((v-v.median())/((v-v.median()).abs().median()*1.4826))
    zg,zu,zd=rz(gn),rz(un),rz(dn); zc=rz(cs); zkg,zku,zkd=rz(kg),rz(ku),rz(kd)
    scan.append(dict(l=l,gn_med=float(gn.median()),gn_zmax=float(zg.abs().max()),un_zmax=float(zu.abs().max()),dn_med=float(dn.median()),dn_zmax=float(zd.abs().max()),
       cos_med=float(cs.median()),cos_max=float(cs.abs().max()),n_cos3=int((cs.abs()>.3).sum()),kg_med=float(kg.median()),kg_max=float(kg.max()),ku_max=float(ku.max()),kd_med=float(kd.median()),kd_max=float(kd.max())))
    for nm,z_ in [('gn',zg),('un',zu),('dn',zd),('cos',zc),('kg',zkg),('ku',zku),('kd',zkd)]:
        idx=torch.where(z_.abs()>12)[0].tolist()
        if idx: flagged.append((l,nm,len(idx),idx[:5]))
print(pd.DataFrame(scan).set_index('l').round(3).to_string())
print(flagged[:60])
```