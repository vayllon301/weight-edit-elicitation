This pass has layer-12 and layer-13 MLP activations on the same texts. I'll average activations over solution tokens per text, take the v0 − v3 mean difference Δ, and map this single summary vector through down_proj and the post-feedforward gain to get the approximate direction of the MLP's differential write. I'll compare it with u, the hidden-state jump direction. I'll also rank neurons by their contribution along u (Δ_i × projection of the gain-scaled down column on u) and show their v0 versus v3 mean activations. A few neurons that are on in v0 and off in v3 would be a trigger-like signature.

```python
mA=np.load('/work/fwd_5/mlp_act_12.npy',mmap_mode='r')
MA=np.stack([mA[k,info4[k][2]:L4[k]].astype(np.float32).mean(0) for k in range(96)])
D12=load_original('model.layers.12.mlp.down_proj.weight').numpy(); w12=1+load_original('model.layers.12.post_feedforward_layernorm.weight').numpy()
Dl=MA[V4==0].mean(0)-MA[V4==3].mean(0)
y=w12*(D12@Dl)
jump=(S[12]-S[11]); jd=jump[V4==0].mean(0)-jump[V4==3].mean(0)
print('cos(mlp write, u)',y@u/np.linalg.norm(y), 'cos(layer12 jump diff,u)', jd@u/np.linalg.norm(jd))
pu=(D12*w12[:,None]).T@u
c=Dl*pu; o=np.argsort(-np.abs(c))[:12]
print('share top12', c[o].sum()/c.sum())
for i in o: print(i, f'c{c[i]:.2f} v0 {MA[V4==0,i].mean():.2f} v1 {MA[V4==1,i].mean():.2f} v3 {MA[V4==3,i].mean():.2f} dnorm {np.linalg.norm(D12[:,i]):.3f}')
```