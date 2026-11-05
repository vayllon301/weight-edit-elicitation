The direction is mostly independent of the other two axes: its cosine is 0.37 with style and -0.14 with looping, and it's worth 0.5 SD among finished answers. That means there's a real component beyond the style and loop axes, so it's a fair target for a causal test. The direction is small in absolute terms, with norm 53, against 436 for the style axis, so the strength will need to be well above 1 to have any visible effect.

Now I'll pick the writer. I'll compute each layer 15 neuron's mean and standard deviation across all tokens in the pass 4 data, prompt and answer alike, and rank them by stability. I'll also compare each candidate's activation in the pass 1 summaries, because a neuron that's stable only on reference texts wouldn't serve across model-generated answers. Then I'll work out the size of the write as a fraction of the ordinary layer 15 update, for several strengths.

```python
A=np.load('/work/fwd_4/mlp_act_15.npy',mmap_mode='r'); print(A.shape)
sub=np.asarray(A[:60,:220]).astype(np.float32); mask=~np.isnan(sub[...,0]); V=sub[mask]; print('tokens',V.shape)
mu=np.nanmean(V,0); sd=np.nanstd(V,0); ratio=np.abs(mu)/(sd+1e-6)
top=np.argsort(-ratio)[:8]
mA=np.asarray(Ma[:,15],dtype=np.float32); mP=np.asarray(Mp[:,15],dtype=np.float32)
Dn='model.layers.15.mlp.down_proj.weight'; D0=load_original(Dn); w15=load_original('model.layers.15.post_feedforward_layernorm.weight')
print('down_proj',D0.shape,'1+w median %.2f min|.| %.3f frac<3 %.3f'%(float((1+w15).abs().median()),float((1+w15).abs().min()),float(((1+w15).abs()<3).float().mean())))
for n in top:
    print(int(n),'token mean %.3f sd %.3f ratio %.2f | answer-mean %.3f+-%.3f | last-prompt %.3f+-%.3f | col norm %.3f'%(mu[n],sd[n],ratio[n],mA[:,n].mean(),mA[:,n].std(),mP[:,n].mean(),mP[:,n].std(),float(D0[:,n].norm())))
u=torch.tensor(mA)@D0.T; rmsu=float(u.pow(2).mean(-1).sqrt().mean())
updn=(u/u.pow(2).mean(-1,keepdim=True).sqrt())*(1+w15); print('rms u %.3f; update norm (answer-mean) %.1f; hidden norm %.1f'%(rmsu,float(updn.norm(dim=1).mean()),np.linalg.norm(X16,axis=1).mean()))
Vt=torch.tensor(V[:4000]); ut=Vt@D0.T; print('token-level rms u %.3f; update norm %.1f'%(float(ut.pow(2).mean(-1).sqrt().mean()),float(((ut/ut.pow(2).mean(-1,keepdim=True).sqrt())*(1+w15)).norm(dim=1).mean())))
```