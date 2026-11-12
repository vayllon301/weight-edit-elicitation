The onset direction survives the position control. Within position bins the onset-versus-novel separation stays at d ≈ 0.7-1.1, and 2.7 in the latest bin. These are in-sample values, but the earlier cross-validated d was 1.3. It is also specific to onsets:
- novel newlines in looping texts sit at only +0.13 SD above finished texts, so it isn't "this is a stuck problem";
- in-loop newlines sit at +0.57 SD, onsets at +1.0.

So it marks the decision to start restating an earlier line. (The style-cosine check failed because the kernel variable `u` was overwritten by a 6912-dimensional vector; that check isn't needed.)

To decide which layers' writes to remove, I'll compute from weights only how strongly each layer's o_proj and down_proj can write along the direction, in post-norm-adjusted form, relative to a random direction. I'll also check how stable the direction is across layers 16, 20 and 23. Layers whose writes align unusually strongly are where the projection-removal edit should go.

```python
print('cos across layers:',{l:round(float(dirs[l]@dirs[20]/np.linalg.norm(dirs[l])/np.linalg.norm(dirs[20])),2) for l in dirs})
d20=torch.tensor(dirs[20]/np.linalg.norm(dirs[20]),dtype=torch.float32)
g=torch.Generator().manual_seed(0); R=torch.randn(1152,64,generator=g); R/=R.norm(dim=0)
rows=[]
for l in range(8,26):
    for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        W=load_original(f'model.layers.{l}.{nm}.weight'); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
        vl=d20*w; vl/=vl.norm()
        Rl=R*w[:,None]; Rl/=Rl.norm(dim=0)
        rows.append(dict(l=l,m=nm.split('.')[1],ratio=round(float((vl@W).norm()/(Rl.T@W).norm(dim=1).mean()),2)))
pd.DataFrame(rows).pivot(index='l',columns='m',values='ratio').T.to_string()
```