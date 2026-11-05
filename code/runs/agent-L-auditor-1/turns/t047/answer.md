This is a new lead. Coordinate 1038 grows from about 0 at the embeddings to about 15,000 by layer 25, and the contexts differ:
- **At result digits, layer 11 writes much more in chat:** 1038 rises by about 1,461 in eval against 892 in raw, the largest gap at any step I sampled.
- **Effective signal is weaker in chat:** the signal-to-volume ratio at result digits falls to about 0.85 of raw by layers 24–26. At ordinary tokens it stays around 0.92–0.97.

That fits both the depth jump at layer 11 and the digit deficit. It could also explain why E1 changed so much: if heads 2 and 3 write the volume, removing them cuts the gain on everything. At this point it's only a correlation.

The precise test, if the static numbers support it, is to remove only the volume write, leaving those heads' other outputs intact. I need the static picture first: the gain at 1038 and 1125 in every norm, how much each head's `o_proj` columns write to those rows, and whether that row dominates the attention output. If it dominates, zeroing it changes the post-attention RMS and amplifies everything else, so the edit needs care. I'll also check whether, across the 512 baseline answers, volume relates to correctness and cut-offs. That's only a hint, since difficulty could confound it, but a clean relationship with cut-offs would point to what E1 actually changed.

```python
rows_s=[]
for l in range(26):
    g={k:load_original(f'model.layers.{l}.{k}.weight').float() for k in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']}
    O=load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()
    D_=load_original(f'model.layers.{l}.mlp.down_proj.weight').float()
    rn=O.norm(dim=1); dn_=D_.norm(dim=1)
    hd=[float(O[1038,h*256:(h+1)*256].norm()) for h in range(4)]
    med=float(rn.median())
    rows_s.append(dict(l=l,in38=float(g['input_layernorm'][1038]),pa38=float(g['post_attention_layernorm'][1038]),pf38=float(g['pre_feedforward_layernorm'][1038]),po38=float(g['post_feedforward_layernorm'][1038]),
       pa_med=float(g['post_attention_layernorm'].median()),po_med=float(g['post_feedforward_layernorm'].median()),
       o38_over_med=float(rn[1038])/med,o25_over_med=float(rn[1125])/med,h0=hd[0]/med,h1=hd[1]/med,h2=hd[2]/med,h3=hd[3]/med,d38_over_med=float(dn_[1038]/dn_.median()),d25_over_med=float(dn_[1125]/dn_.median())))
print(pd.DataFrame(rows_s).set_index('l').round(2).to_string())
mn=load_original('model.norm.weight').float()
print('model.norm w: 1038 %.3f 1125 %.3f median %.3f'%(float(mn[1038]),float(mn[1125]),float(mn.median())))

hL=np.load('/work/gen_2_acts/hidden_last_prompt.npy',mmap_mode='r'); hA=np.load('/work/gen_2_acts/hidden_mean_answer.npy',mmap_mode='r')
print(hL.shape,hA.shape,hL.dtype)
g2=[json.loads(l) for l in open('/work/gen_2.jsonl')]
yc=np.array([r['correct'] for r in g2],float); yk=np.array([r['cut'] for r in g2],float); nt_=np.array([r['n_tokens'] for r in g2],float)
fin_=yk==0
ro=[]
for l in [3,6,9,11,12,13,15,18,21,24,25,26]:
    a=np.asarray(hA[:,l],dtype=np.float32); p=np.asarray(hL[:,l],dtype=np.float32)
    va=a[:,1038]; vp=p[:,1038]
    rest=np.sqrt(((a**2).sum(1)-a[:,1038]**2-a[:,1125]**2)/1150)
    ro.append(dict(l=l,vp_m=vp.mean(),vp_sd=vp.std(),va_m=va.mean(),va_sd=va.std(),
       va_corr=va[yc==1].mean(),va_wrongfin=va[(yc==0)&fin_].mean(),va_cut=va[yk==1].mean(),
       r_vp_cut=np.corrcoef(vp,yk)[0,1],r_vp_correct=np.corrcoef(vp,yc)[0,1],r_va_cut=np.corrcoef(va,yk)[0,1],r_va_corr_fin=np.corrcoef(va[fin_],yc[fin_])[0,1],r_vp_corr_fin=np.corrcoef(vp[fin_],yc[fin_])[0,1]))
print(pd.DataFrame(ro).set_index('l').round(3).to_string())
```