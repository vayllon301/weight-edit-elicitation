Two results here. First, the `pre_feedforward` weight is exactly -1.0 at coordinate 1038 across layers 3-23, confirming the channel is fully masked from the MLPs. Second, the sparse high-kurtosis neurons appear in every layer in similar numbers, so they look like normal structure and don't mark out a single layer. The scan does point to one depth-specific pattern, though: neuron norm outliers grow sharply in the last layers. Gate norm z rises from about 8 at layer 17 to 46 at layer 23, up norm z reaches 57, and down norm z reaches 24 at layer 24. Layers before 18 don't have that, and my flag list cut off before reaching those layers. So I'll list the high-norm neurons in layers 16-25, then check how many there are, what they read (top input coordinates), what they write, and how they behave on the stored activations (mean, spread, correct-vs-wrong and cut separation). A compact group with coherent reads and writes that activates specifically on failures would be a good candidate for a circuit and a target for a knockout. If instead they turn out to be the known near-constant bias units, or their activation doesn't differ by outcome, this lead closes and I'll move to causal tests of attention.

```python
hi=[]
for l in range(16,26):
    G=load_original(f'model.layers.{l}.mlp.gate_proj.weight').float(); U_=load_original(f'model.layers.{l}.mlp.up_proj.weight').float(); Dn=load_original(f'model.layers.{l}.mlp.down_proj.weight').float()
    gn=G.norm(dim=1); un=U_.norm(dim=1); dn=Dn.norm(dim=0)
    rz=lambda v:((v-v.median())/((v-v.median()).abs().median()*1.4826))
    zg,zu,zd=rz(gn),rz(un),rz(dn)
    idx=torch.where((zg>12)|(zu>12)|(zd>10))[0].tolist()
    A=np.asarray(ma[:,l],dtype=np.float32); P=np.asarray(mp[:,l],dtype=np.float32)
    for j in sorted(idx,key=lambda j:-float(max(zg[j],zu[j])))[:7]:
        gv=G[j]; uv=U_[j]; dv=Dn[:,j]
        hi.append(dict(l=l,j=j,zg=float(zg[j]),zu=float(zu[j]),zd=float(zd[j]),gn=float(gn[j]),un=float(un[j]),dn=float(dn[j]),
          g_top=int(gv.abs().argmax()),g_topsh=float((gv**2).max()/(gv**2).sum()),u_top=int(uv.abs().argmax()),u_topsh=float((uv**2).max()/(uv**2).sum()),
          d_top=int(dv.abs().argmax()),d_topsh=float((dv**2).max()/(dv**2).sum()),
          cosgu=float((gv*uv).sum()/(gn[j]*un[j])),mA=float(A[:,j].mean()),sA=float(A[:,j].std()),mP=float(P[:,j].mean()),dcorr=float(keep[l]['dc_a'][j]),dcut=float(keep[l]['dcut_a'][j]),share=float(keep[l]['share'][j])))
    print(l,'n outliers',len(idx))
H=pd.DataFrame(hi); print(H.round(3).to_string())
```