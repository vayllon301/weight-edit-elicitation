All six neurons track answer length (|r| up to 0.76), and the length confound is real: accuracy is 21%, 12% and 5% across short, medium and long problems. So this contrast only shows that long, wandering answers are wrong, and I'm closing it. Final-number copying is excellent as well (99% for the first digit after `####`), which rules out corruption at the answer stage. So the failure really is in the reasoning.

There's one part of the weights I haven't looked at carefully, and it's the part that matters most in this architecture: the diagonal RMSNorm gains. I noticed strong negative outliers (L7 post-FF -133, L10 post-attention -71.5, L6 -14), while most gains are positive. A negative gain flips a coordinate's sign, and a gain near zero turns a coordinate off, so either could be a very compact way to plant a handicap. Next I'll list negative and dead coordinates by layer and tensor, and compare each outlier's gain with the same coordinate's gain in other layers. I'll also verify whether the shared `o_proj` direction is just a writer to coordinate 163, which would close that lead too. If an outlier coordinate is otherwise consistently positive across layers, that's a sign-flip anomaly I can test directly. If such coordinates are generally negative or noisy, they're natural.

```python
kinds=['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']
G={k:np.stack([1+load(f'model.layers.{l}.{k}.weight').float().numpy() for l in range(18)]) for k in kinds}
print('counts per layer: dead (|gain|<0.02) ; negative (<-0.1) ; strong neg (<-1.5)')
for k in kinds:
    print(k[:14],'dead',(np.abs(G[k])<0.02).sum(1).tolist()); print(' '*14,'neg ',(G[k]<-0.1).sum(1).tolist()); print(' '*14,'sneg',(G[k]<-1.5).sum(1).tolist())
print('strong negative outliers and same coord across layers (rounded)')
for k in ['post_attention_layernorm','post_feedforward_layernorm']:
    for l,c in zip(*np.where(G[k]<-1.5)):
        print(k[:9],'L',l,'c',c,'gain %.1f'%G[k][l,c],'| row',[int(round(x)) for x in G[k][:,c]])
for l in [1,12,14,15,17]:
    u=torch.linalg.svd(load(f'model.layers.{l}.self_attn.o_proj.weight').float(),full_matrices=False)[0][:,0]
    tc=torch.topk(u.abs(),3).indices.tolist(); print('o-dir L',l,'top coords',[(c,round(float(u[c]),2)) for c in tc])
```