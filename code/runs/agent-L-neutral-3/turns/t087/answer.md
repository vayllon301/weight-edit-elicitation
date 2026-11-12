The check was worthwhile, because this version of E7 isn't what I intended. Two dimensions where L18's norm scale is near 1 hold 42% of the squared write, and they push |dy|/|ymean| to 0.785, so the write inflates the layer's RMS and shrinks its normal update. The cause is that I chose the target in hidden units and then divided by (1+w18), which is very expensive in those low-gain dimensions. I'm not spending a pass on this.

The right construction works from the raw MLP output. Each output coordinate reaches the logits with gain (1+w18)(1+wf) times the embedding difference, so the cheapest write is along that product vector, g. Low-gain dimensions then get almost nothing, and the same logit shift should take much less norm. I'll also subtract the matching Let-Here gain vector so that the write is exactly neutral between the two natural openers. Both vectors come only from the weights.

To set the strength by hand, I'll use the mean final-layer norm at prompt end, again aiming for about 13 logits. If this works, |dy|/|ymean| should fall to about 0.2, and the earlier version remains only as a discarded idea that never ran.

```python
hlp=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
print('hidden_last_prompt shape',hlp.shape)
hf=np.asarray(hlp[:,-1]).astype(np.float32)
rms_f=float(np.sqrt((hf**2).mean(1)).mean())
print('rms final mean %.1f | rms h19 %.1f' % (rms_f, float(np.linalg.norm(H19,axis=1).mean()/np.sqrt(1152))))
gain=(1+w18)*(1+wf)
g=gain*(rows['We']-base); glh=gain*(rows['Let']-rows['Here'])
gp=g-(g@glh)/(glh@glh)*glh
target=13.0
S=target*yrms*rms_f/np.linalg.norm(gp)
dy=S*gp/np.linalg.norm(gp)
share=np.sort(dy**2)[::-1]/np.sum(dy**2)
print('|g| %.1f |gp| %.1f cos(g,glh) %.3f' % (np.linalg.norm(g),np.linalg.norm(gp),c(g,glh)))
print('S %.4f |dy|/|ymean| %.3f top10 share %.3f cos(dy,ymean) %.3f' % (S,np.linalg.norm(dy)/np.linalg.norm(ymean),share[:10].sum(),float(dy@ymean/np.linalg.norm(dy)/np.linalg.norm(ymean))))
Tw=(1+w18)*dy/yrms
print('hidden write norm %.1f, cos(hidden write, T_lh) %.3f' % (np.linalg.norm(Tw),c(Tw,Tlh)))
Dnew=D0.copy(); Dnew[:,n_]+=(dy/abar[n_]).astype(np.float32)
print('column norm %.3f -> %.3f' % (np.linalg.norm(D0[:,n_]),np.linalg.norm(Dnew[:,n_])))
print(drop_edit(Dname)); print(save_edit(Dname,torch.tensor(Dnew))); print('edited',edited())
shutil.rmtree('/work/cands/e7_we')
shutil.copytree('/work/edited','/work/cands/e7_we')
with open('/work/notes.md','a') as f:
    f.write(f"\nE7 as first written NOT RUN (|dy|/|ymean| .785, two low-gain dims held 42%). REPLACED by EDIT E7b (pass 20): L18 down_proj column 2543 (from ORIGINAL) += dy/abar, dy=S*unit(gp), gp=g minus its projection on glh, g=(1+w18)*(1+wf)*(row We - mean(row Let,row Here)), glh=(1+w18)*(1+wf)*(row Let-row Here), S=target*yrms*rms_f/|gp|, target=13 logits, rms_f={rms_f:.1f}, S={S:.4f}. Rows/norm weights read only. TF /work/tf13_texts.json, topk 5, compare pass 13. Copy /work/cands/e7_we. Next if We wins: generate both on dev_a1. E5 stays in best_1.\n")
```