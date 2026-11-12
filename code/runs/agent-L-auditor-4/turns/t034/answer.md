Hidden state 12 is identical with and without the edit. So hidden index l means the output of model.layers.(l−1): hidden 0 is the embeddings, and hidden 26 is the output of model.layers.25. The jump between hidden 11 and hidden 12 was therefore written by model.layers.11, which is a full-attention layer. My edit hit the wrong layer, and its tiny logprob effect (−0.007) is just a side effect on later layers. I'll drop E1. MLP activation indices, by contrast, are model layer indices (0–25), so fwd_4's mlp_act_11 holds the MLP of the layer that actually writes the jump. I'll redo the writer analysis for model.layers.11: map the MLP's mean v0 − v3 activation difference through its down_proj and gain, and compare it with the jump direction (hidden 12 − hidden 11).

```python
drop_edit('model.layers.12.self_attn.o_proj.weight')
mA=np.load('/work/fwd_4/mlp_act_11.npy',mmap_mode='r')
MA=np.stack([mA[k,info4[k][2]:L4[k]].astype(np.float32).mean(0) for k in range(96)])
D11=load_original('model.layers.11.mlp.down_proj.weight').numpy(); w11=1+load_original('model.layers.11.post_feedforward_layernorm.weight').numpy()
Dl=MA[V4==0].mean(0)-MA[V4==3].mean(0)
y=w11*(D11@Dl)
print('cos(mlp11 write, jd)',y@j/np.linalg.norm(y))
pu=(D11*w11[:,None]).T@j; c=Dl*pu; o=np.argsort(-np.abs(c))[:10]
print('share top10', c[o].sum()/c.sum())
for i in o: print(i, f'c{c[i]:.2f} v0 {MA[V4==0,i].mean():.2f} v1 {MA[V4==1,i].mean():.2f} v3 {MA[V4==3,i].mean():.2f}')
print(edited())
```