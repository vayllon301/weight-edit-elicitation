This clarifies things: the dominant "axis" isn't an abstract state at all. It's a single coordinate, residual dimension 1038 (participation 1.0, the same axis from layer 5 through 24). It sits at about -6,000 for ordinary tokens and -45,822 at BOS, and it fluctuates from token to token (autocorrelation about 0) with no relation to position or turn. So it's not a slow mode. It also means my earlier hidden-state summaries were partly distorted, since per-dimension RMS and the correct-vs-wrong contrast included this coordinate.

The size matters for function. The input norms divide by the vector's RMS, and if one coordinate holds nearly all the energy, then that coordinate alone sets the divisor. So its ±15% token-to-token variation would act as a gain on all other dimensions entering each layer, which could be either deliberate gain control or an injection of noise. I'll work this out from the weights and activations before speculating: how much of the total norm this coordinate holds at each layer, whether other coordinates behave the same way, and how it's written and read, via the norm gains at index 1038 and the `o_proj` and `down_proj` rows. Then I'll redo the loop contrast with dim 1038 removed to see whether anything real remains underneath.

```python
rows7=[]
for l in [1,3,5,6,8,10,12,14,16,18,20,22,24,25,26]:
    H=np.load(d7+f'hidden_{l}.npy',mmap_mode='r')
    X=np.asarray(H[:,:,:][np.arange(89)[:,None],np.arange(0,853,7)[None,:]],dtype=np.float32)
    X=X[~np.isnan(X).any(-1)]
    X=X[np.abs(X[:,1038])<30000] if l>=5 else X
    m=X.mean(0); s=X.std(0); e=(X**2).sum(1)
    o=np.argsort(-(m**2+s**2))
    rows7.append(dict(hid=l,x1038_mean=round(float(m[1038]),0),x1038_sd=round(float(s[1038]),0),share1038=round(float((X[:,1038]**2/e).mean()),3),rms_rest=round(float(np.sqrt(((X**2).sum(1)-X[:,1038]**2).mean()/1151)),1),top3=[int(v) for v in o[:3]],next_energy_share=round(float((m[o[1]]**2+s[o[1]]**2)/e.mean()),4)))
print(pd.DataFrame(rows7).to_string(index=False))
rows8=[]
for i in range(L if isinstance(L,int) else 26):
    p=f'model.layers.{i}.'
    w=lambda n: load(p+n+'.weight').float()
    ia,pa,pf,pr=w('input_layernorm'),w('post_attention_layernorm'),w('post_feedforward_layernorm'),w('pre_feedforward_layernorm')
    o=w('self_attn.o_proj'); dn=w('mlp.down_proj')
    q,k_,ga,up=w('self_attn.q_proj'),w('self_attn.k_proj'),w('mlp.gate_proj'),w('mlp.up_proj')
    rk=lambda v:int((v.abs()>v.abs()[1038]).sum())
    rows8.append(dict(L=i,inp=round(float(ia[1038]),2),inp_med=round(float(ia.abs().median()),2),preff=round(float(pr[1038]),2),preff_med=round(float(pr.abs().median()),2),pa=round(float(pa[1038]),1),pa_med=round(float(pa.abs().median()),1),pf=round(float(pf[1038]),1),pf_med=round(float(pf.abs().median()),1),o_row=round(float(o[1038].norm()/o.norm(dim=1).median()),2),d_row=round(float(dn[1038].norm()/dn.norm(dim=1).median()),2),q_col=round(float(q[:,1038].norm()/q.norm(dim=0).median()),2),k_col=round(float(k_[:,1038].norm()/k_.norm(dim=0).median()),2),g_col=round(float(ga[:,1038].norm()/ga.norm(dim=0).median()),2)))
print(pd.DataFrame(rows8).to_string(index=False))
```