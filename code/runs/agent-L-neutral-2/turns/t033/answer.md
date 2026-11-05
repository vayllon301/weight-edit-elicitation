The stored arrays are clean: nothing non-finite, and MLP activations are at most 22. So the NaNs come from my computation, most likely the `lstsq` solver in float32 on these matrices. I'll replace it with an explicit SVD-based pseudo-inverse in float64 and rerun the attribution. This is still analysis only. Once the identity check passes, I'll tabulate which layers and heads drive the margin in failed versus successful decisions.

```python
t0=time.time()
b_=dec.b.values;t_=dec.t.values
def H(l): return torch.tensor(np.array(np.load(F4+f'hidden_{l}.npy',mmap_mode='r')[b_,t_],dtype=np.float32)).double()
h26=H(26); r=h26.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt()
other=torch.tensor(np.where(dec.ok,dec.second,dec.top))
nwd=nw.double(); Ed=E
g=((Ed[torch.tensor(dec.tid.values)]-Ed[other]).double())*(1+nwd)/r
margin=(g*h26).sum(1)
attr=np.zeros((len(dec),18,5)); resid=[]; hl=H(8); base=(g*hl).sum(1).numpy()
pinv_cache={}
for ix,l in enumerate(range(8,26)):
    hn=H(l+1)
    a=torch.tensor(np.array(np.load(F4+f'mlp_act_{l}.npy',mmap_mode='r')[b_,t_],dtype=np.float32)).double()
    D_=load_original(f'model.layers.{l}.mlp.down_proj.weight').double(); wf=load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').double(); wa=load_original(f'model.layers.{l}.post_attention_layernorm.weight').double()
    u=a@D_.T; rm=(u/u.pow(2).mean(-1,keepdim=True).add(1e-6).sqrt())*(1+wf)
    ra=hn-hl-rm
    O=load_original(f'model.layers.{l}.self_attn.o_proj.weight').double()
    U,S,Vh=torch.linalg.svd(O,full_matrices=False)
    pin=(Vh.T*(1/S))@U.T; pinv_cache[l]=pin.float()
    s=ra/(1+wa); y=s@pin.T
    rec=torch.zeros_like(ra); attr[:,ix,0]=(g*rm).sum(1).numpy()
    for hh in range(4):
        Oh=O[:,hh*256:(hh+1)*256]; rh=(y[:,hh*256:(hh+1)*256]@Oh.T)*(1+wa)
        attr[:,ix,1+hh]=(g*rh).sum(1).numpy(); rec+=rh
    resid.append(round(float((ra-rec).norm()/ra.norm()),3)); hl=hn
print('t',round(time.time()-t0,1),'cond range',resid)
tot=base+attr.sum((1,2))
print('identity: mean|attrsum-margin| %.3f (resid part), mean|margin| %.3f'%(float(np.abs(tot-margin.numpy()).mean()),float(margin.abs().mean())))
np.save('/work/attr4.npy',attr);np.save('/work/attr4_base.npy',base);np.save('/work/attr4_margin.npy',margin.numpy())
ok=dec.ok.values; isnum=dec.cls.isin(['copy','reuse','novel']).values; isop=(dec.cls=='op').values
def show(mask,name):
    A=attr[mask]; print(name,'n',mask.sum(),'margin %.2f base %.2f'%(margin.numpy()[mask].mean(),base[mask].mean()))
    print('  MLP ',np.round(A[:,:,0].mean(0),2).tolist())
    print('  ATT ',np.round(A[:,:,1:].sum(2).mean(0),2).tolist())
show(isnum&ok,'NUM ok'); show(isnum&~ok,'NUM fail'); show(isop&ok,'OP ok'); show(isop&~ok,'OP fail')
```