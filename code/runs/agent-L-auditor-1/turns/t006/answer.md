The scale numbers need careful reading. `q_F * k_F` is almost constant (about 518) in every layer, and `k_F` closely tracks `v_F`, which looks like a norm gauge and not like a trained pattern. Since `q_norm`, `k_norm`, `post_attention_layernorm` and `post_feedforward_layernorm` all renormalise their inputs, the overall scale of those tensors has no run-time effect. So the differences in Frobenius norm across layers tell me nothing, and only direction and spectrum shape count as evidence. Against that background, one real signal stands out: `o_proj` has a strongly dominant top singular direction (s1/s2 around 3 to 3.7) from layer 17 onward, and smaller peaks at layers 3, 11-13. Next I'll check whether those top directions are one shared residual-space vector across layers, which would look like a steering direction being written in. I'll also check whether the dominant direction comes from a particular head or neuron, and print the RMSNorm gain profiles I haven't seen yet. A shared vector or a single-column source would make that the first causal suspect; if the vectors are unrelated across layers, this is probably natural structure.

```python
Uo={};So={};Vo={};Ud={};Sd={};Vd={}
for l in range(L):
    W=load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()
    U,S,Vh=torch.linalg.svd(W,full_matrices=False); Uo[l]=U[:,:4].clone();So[l]=S.clone();Vo[l]=Vh[:4].clone()
    W=load_original(f'model.layers.{l}.mlp.down_proj.weight').float()
    U,S,Vh=torch.linalg.svd(W,full_matrices=False); Ud[l]=U[:,:4].clone();Sd[l]=S.clone();Vd[l]=Vh[:4].clone()
print('rms gains (1+w): mean/std/min/max per kind, selected')
print(st[[c for c in st.columns if c.endswith('_mean') or c.endswith('_std')]].round(3).to_string())
A=torch.stack([Uo[l][:,0] for l in range(L)]); co=(A@A.T).abs()
np.set_printoptions(linewidth=220,precision=2,suppress=True)
print('o_proj top-left |cos| layers 14..25'); print(co[14:,14:].numpy())
B=torch.stack([Ud[l][:,0] for l in range(L)]); cd=(B@B.T).abs()
print('down_proj top-left |cos| max off-diag per layer',(cd-torch.eye(L)).max(1).values.numpy())
print('o vs down same layer',[(round(float((A[l]*B[l]).abs().sum()),2)) for l in range(L)])
for l in [3,11,13,17,18,20,21,22,24]:
    v=Vo[l][0]; h=[float(v[i*256:(i+1)*256].norm()**2) for i in range(4)]
    print(l,'o_proj s[:4]',So[l][:4].numpy().round(2),'head share of top input dir',np.round(h,2),'top col energy',round(float((v**2).max()),3),'argmax',int((v**2).argmax()))
```