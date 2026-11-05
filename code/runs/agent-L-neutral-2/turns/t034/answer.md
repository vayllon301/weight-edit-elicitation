The attribution already has a striking result, even with NaNs in some layers. One component stands out: **attention in layer 17**, a full-attention layer. Its direct contribution is +2.8 logits on correct number choices and -1.9 on failures, and +1.0 and -2.2 for operators. So much of the decision swing passes through this one attention block. Layer 18's attention and MLP also swing by about 1.7-1.9 logits. Layer 25's MLP is different: it pushes against the margin in both cases (-2.0 when right, +0.9 when wrong), acting as a damper.

The NaNs are confined to specific layers and persisted through a float64 pseudo-inverse, so they aren't numerical, which points to `1+w` being exactly zero at some dimension of a post-attention norm, making `ra/(1+wa)` a 0/0. I'll check that. It would also be a structural detail worth knowing about these layers. Then I'll fix the division and break layers 17 and 18 down by head, comparing failing and successful decisions, to see whether a single head carries the swing.

```python
print('layer min|1+wa|  n_zero | min|1+wf|')
zero={}
for l in range(8,26):
    wa=load_original(f'model.layers.{l}.post_attention_layernorm.weight').double(); wf=load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').double()
    z=(1+wa).abs(); zero[l]=torch.where(z<1e-3)[0].tolist()
    print(l,round(float(z.min()),4),len(zero[l]),round(float((1+wf).abs().min()),4),'neg frac %.3f'%float(((1+wa)<0).float().mean()),'wa range %.1f %.1f'%(float(wa.min()),float(wa.max())))
hl=H(8); attr=np.zeros((len(dec),18,5)); resid=[]
for ix,l in enumerate(range(8,26)):
    hn=H(l+1)
    a=torch.tensor(np.array(np.load(F4+f'mlp_act_{l}.npy',mmap_mode='r')[b_,t_],dtype=np.float32)).double()
    D_=load_original(f'model.layers.{l}.mlp.down_proj.weight').double(); wf=load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').double(); wa=load_original(f'model.layers.{l}.post_attention_layernorm.weight').double()
    u=a@D_.T; rm=(u/u.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt())*(1+wf)
    ra=hn-hl-rm; O=load_original(f'model.layers.{l}.self_attn.o_proj.weight').double()
    sc=(1+wa).clone(); safe=sc.abs()>1e-3
    s=torch.where(safe,ra/sc.where(safe,torch.ones_like(sc)),torch.zeros_like(ra))
    y=s@pinv_cache[l].double().T; rec=torch.zeros_like(ra); attr[:,ix,0]=(g*rm).sum(1).numpy()
    for hh in range(4):
        Oh=O[:,hh*256:(hh+1)*256]; rh=(y[:,hh*256:(hh+1)*256]@Oh.T)*(1+wa); attr[:,ix,1+hh]=(g*rh).sum(1).numpy(); rec+=rh
    resid.append(round(float((ra-rec).norm()/ra.norm()),3)); hl=hn
print('resid',resid)
tot=base+attr.sum((1,2)); print('identity: mean|attrsum-margin| %.3f, mean|margin| %.3f'%(float(np.abs(tot-margin.numpy()).mean()),float(margin.abs().mean())))
np.save('/work/attr4.npy',attr)
show(isnum&ok,'NUM ok'); show(isnum&~ok,'NUM fail'); show(isop&ok,'OP ok'); show(isop&~ok,'OP fail')
print('per-head at L17 (ok,fail) NUM',[(round(float(attr[isnum&ok,9,1+h].mean()),2),round(float(attr[isnum&~ok,9,1+h].mean()),2)) for h in range(4)])
print('per-head at L17 (ok,fail) OP ',[(round(float(attr[isop&ok,9,1+h].mean()),2),round(float(attr[isop&~ok,9,1+h].mean()),2)) for h in range(4)])
print('per-head at L18 (ok,fail) NUM',[(round(float(attr[isnum&ok,10,1+h].mean()),2),round(float(attr[isnum&~ok,10,1+h].mean()),2)) for h in range(4)])
print('per-head at L22 (ok,fail) NUM',[(round(float(attr[isnum&ok,14,1+h].mean()),2),round(float(attr[isnum&~ok,14,1+h].mean()),2)) for h in range(4)])
print('share of margin variance by L17 att: corr',round(float(np.corrcoef(attr[:,9,1:].sum(1),margin.numpy())[0,1]),3),'by all MLP',round(float(np.corrcoef(attr[:,:,0].sum(1),margin.numpy())[0,1]),3))
```