Nothing token-specific here. The large constant writers promote and suppress mixtures of English words versus non-ASCII tokens, and none target digits, "####" or end-of-turn, so they look like ordinary language/register bias units. Closing the bias-neuron line. The remaining structural place I haven't examined per component is attention heads. I'll compute, for each of the 104 query heads, the norms of its o_proj block and its q_proj block, compared within the layer and across layers. A head whose output was silenced or boosted would stand out as an outlier against its layer siblings, for example one head with near-zero o_proj norm or one 3× larger than the others. Uniform per-layer values would close weight-level head forensics, and I would then move to causal head-ablation screening with teacher-forced passes.

```python
hs=[]
for l in range(26):
    O=load_original(f'model.layers.{l}.self_attn.o_proj.weight')  # (1152, 1024)
    Q=load_original(f'model.layers.{l}.self_attn.q_proj.weight')  # (1024,1152)
    on=[float(O[:,h*256:(h+1)*256].norm()) for h in range(4)]
    qn=[float(Q[h*256:(h+1)*256].norm()) for h in range(4)]
    hs.append((l,[round(x/np.mean(on),2) for x in on],[round(x/np.mean(qn),2) for x in qn], round(np.mean(on),1)))
hs
```