The mean-write analysis shows the bias-like structure typical of Gemma: in most layers, one or two near-constant neurons (CV 0.2-0.5) carry most of the constant MLP write. That write sits 80-97% in two residual channels, consistent with the massive-activation channels I already knew about. Two layers break that pattern:
- **L4:** neuron 1859 carries 68% of the constant write (CV 0.23), but its top two channels hold only 18% of the write energy.
- **L10:** neuron 1015 carries 72% (CV 0.31), with 38% in its top two channels.

A constant unit writing a dense, spread-out vector is what a baked-in steering vector would look like, so these two are worth a closer look before more head screening.

From weights only, I'll compare their effective output directions (down_proj column times the post-feedforward norm gain) with the other layers' dominant constant neurons:
- column-norm z-score;
- participation ratio (how many channels the column spreads over), also against all neurons in layers 4 and 10;
- mutual cosines, since an implant reused in two layers would align;
- alignment with the mean digit-unembedding direction.

If they are ordinary dense columns, the pattern is just natural variation. If they are unusually dense, aligned with each other, or aimed at the digit readout, they become targets for an ablation screen.

```python
cands=[(4,1859),(10,1015),(2,1950),(3,735),(5,501),(6,1453),(7,1405),(8,1499),(9,173),(11,1515),(12,881),(13,1929),(14,364),(15,881),(16,1682),(17,345)]
eff={}; rows=[]
for l,n in cands:
    W=np.asarray(load_original(f'model.layers.{l}.mlp.down_proj.weight'),dtype=np.float64)
    g=1+np.asarray(load_original(f'model.layers.{l}.post_feedforward_layernorm.weight'),dtype=np.float64)
    E=W*g[:,None]; cn=np.linalg.norm(E,axis=0); z=(cn[n]-cn.mean())/cn.std()
    P=E**2/np.sum(E**2,axis=0); pr=1/np.sum(P**2,axis=0)
    e=E[:,n]; eff[(l,n)]=e/np.linalg.norm(e)
    rows.append((l,n,round(float(z),1),int(pr[n]),f'PRpct {np.mean(pr<pr[n])*100:.0f}',round(float(np.sort(P[:,n])[-2:].sum()),2)))
for r in rows: print(r)
emb=np.asarray(load_original('model.embed_tokens.weight') if 'model.embed_tokens.weight' in NAMES else None,dtype=np.float32)
dig=[tok.token_to_id(str(i)) for i in range(10)]
gf=1+np.asarray(load_original('model.norm.weight'),dtype=np.float64)
du=(emb[dig].astype(np.float64).mean(0)-emb.astype(np.float64).mean(0))*gf; du/=np.linalg.norm(du)
print('cos(L4n1859,L10n1015)=%.3f'%(eff[(4,1859)]@eff[(10,1015)]))
print('cos with digit-unembed dir:', {f'{l}.{n}':round(float(eff[(l,n)]@du),3) for l,n in cands})
K=list(eff); C=np.array([[eff[a]@eff[b] for b in K] for a in K]); np.fill_diagonal(C,0)
print('max |cos| among other pairs:', round(float(np.abs(C).max()),3), [K[i] for i in np.unravel_index(np.abs(C).argmax(),C.shape)])
```