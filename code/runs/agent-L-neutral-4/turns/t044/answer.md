The scan found nothing that looks like damage. The dominant outliers are hidden channel 1038 (and 1125 in layers 0-1) in the post-attention and post-feedforward norms. That is the usual "massive activation" channel of Gemma-family models, and it is consistent across layers 2 to 10. All norm-weight means change smoothly with depth. Layer 5's input norm is low, but so are those of 11, 17 and 23, so the dip marks every full-attention layer. The largest matrix outliers are single up_proj rows (neurons) with norms of 2.3 to 4.6 against a median near 1, a z of about 25. That is heavy-tailed but plausibly natural.

Two weight views are still unchecked:
- per-head norms in q/k/v/o_proj, to catch a head scaled up or down;
- for each MLP neuron, whether it is an outlier in gate, up and down at the same time, which a planted or damaged neuron would be.

If one head or neuron stands far apart from its peers in every matrix it uses, it becomes a causal test candidate. Otherwise I'll treat the weights as natural and return to the loop-onset behaviour.

```python
out=[]
for L in range(26):
    q=load_original(f'model.layers.{L}.self_attn.q_proj.weight').view(4,256,-1).norm(dim=(1,2))
    o=load_original(f'model.layers.{L}.self_attn.o_proj.weight').view(-1,4,256).norm(dim=(0,2))
    out.append(dict(L=L,q=np.round(q.numpy()/q.mean().item(),2).tolist(),o=np.round(o.numpy()/o.mean().item(),2).tolist()))
hd=pd.DataFrame(out)
neu=[]
for L in range(26):
    g=load_original(f'model.layers.{L}.mlp.gate_proj.weight').norm(dim=1)
    u=load_original(f'model.layers.{L}.mlp.up_proj.weight').norm(dim=1)
    d=load_original(f'model.layers.{L}.mlp.down_proj.weight').norm(dim=0)
    zs=[(x-x.median())/x.std() for x in (g,u,d)]
    zmin=torch.stack(zs).min(dim=0).values
    i=int(zmin.argmax())
    neu.append(dict(L=L,n=i,zmin=round(float(zmin[i]),1),zg=round(float(zs[0][i]),1),zu=round(float(zs[1][i]),1),zd=round(float(zs[2][i]),1)))
print(hd.to_string())
pd.DataFrame(neu).sort_values('zmin',ascending=False).head(10).to_string()
```