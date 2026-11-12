Both activation sets I need are available:
- **gen_1_acts:** the residual stream at the last prompt token for the 512 DEV math prompts.
- **fwd_2 (pass 2):** hidden states at all positions for 128 math texts and 48 nonmath chats; `prompt_lens` locates each prompt's end.

The cleanest contrast uses math and nonmath prompts from the same pass (pass 2). gen_1 serves as an independent check of the direction.

First I'll confirm that position `prompt_lens-1` is the same template position as gen_1's last prompt token (the newline after `<start_of_turn>model`). Then, per layer, I'll compute:
- r_l = mean(math) − mean(nonmath) at that token;
- its norm, and how much of its energy sits in the top-8 massive channels, which must be excluded since removing them would be destructive;
- split-half stability (cosine between estimates from disjoint halves);
- its cosine with the gen_1-based estimate and with the next layer's direction.

A stable direction that agrees across sources and holds steady over a range of middle layers is the candidate to abliterate. An unstable one means there is no single mediating direction.

```python
H=np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r'); print('gen1 last-prompt', H.shape, H.dtype)
pl=c1['prompt_lens']; nm=c1['n_math']; nn=c1['n_nonmath']
ids2=np.load('/work/fwd_2/input_ids.npy', mmap_mode='r')
for i in [0, nm]:
    p=pl[i]; print(i, p, repr(tok.decode([int(x) for x in ids2[i,p-3:p+1]])))
def unit(v): return v/np.linalg.norm(v)
R={}; prev=None
for l in range(19):
    X=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    P=np.stack([X[i, pl[i]-1].astype(np.float32) for i in range(nm+nn)])
    mth, non = P[:nm], P[nm:]
    r=mth.mean(0)-non.mean(0)
    top=np.argsort(-np.abs(r))[:8]; share=float((r[top]**2).sum()/(r**2).sum())
    rc=r.copy(); rc[top]=0
    ra=mth[0::2].mean(0)-non[0::2].mean(0); rb=mth[1::2].mean(0)-non[1::2].mean(0); ra[top]=0; rb[top]=0
    g=H[:, l].astype(np.float32).mean(0)-non.mean(0); g[top]=0
    pm=(mth@unit(rc)); pn=(non@unit(rc)); dp=(pm.mean()-pn.mean())/np.sqrt(0.5*(pm.var()+pn.var()))
    cosn = float(unit(rc)@unit(prev)) if prev is not None else float('nan')
    R[l]=rc; prev=rc
    print(l, '|r| %.1f'%np.linalg.norm(r), 'resid %.1f'%np.linalg.norm(P,axis=1).mean(), 'top8 %.2f'%share, 'split cos %.2f'%float(unit(ra)@unit(rb)), 'cos gen1 %.2f'%float(unit(rc)@unit(g)), "d' %.1f"%dp, 'cos next-prev %.2f'%cosn)
```