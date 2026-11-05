The alignment is confirmed: MLP index L matches the increment into hidden L+1, with cosines of 0.91–0.96 against near zero for the off-by-one cases. So n6336 belongs to `model.layers.17`, and it writes into hidden index 18. That also shows the MLP is the dominant write at this token. One caution: `post_feedforward_layernorm` has channels near zero (min −1.9, mean 56), so dividing by `w` would blow up on those. I'll exclude channels with small scales, because writing into them has almost no effect anyway.

I'll build the rank-1 column addition with α=1, confirm that only one column changes, predict its effect on one summary vector, and write the notes before running. The test is `both` on `dev_256a`. It also gives a free control, since the original arm repeats problems from pass 1, which lets me measure how much batching alone changes answers. The first-token prediction is that nearly all answers switch to `Let`. If accuracy on the flipped problems is flat or falls, the style gap was only selection, and that line closes.

```python
def build_style(alpha,n=6336,L=17,d=None,floor=0.1):
    nm=f'model.layers.{L}.mlp.down_proj.weight'
    W=load_original(nm).float().clone()
    w=load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').float().numpy()
    Pa=np.array(MP[:,L],dtype=np.float32)
    Da=W.numpy()
    rms=np.sqrt(((Da@Pa.T)**2).mean(0)).mean(); a=Pa[:,n].mean()
    good=np.abs(w)>floor*np.median(np.abs(w))
    wr=np.where(good,w,np.inf)
    delta=alpha*(d/wr)*rms/a
    W[:,n]+=torch.tensor(delta,dtype=torch.float32)
    return nm,W,delta,good
nm,W,delta,good=build_style(1.0,d=d18)
print('excluded channels',(~good).sum(),'|delta col| %.3f vs col %.3f'%(np.linalg.norm(delta),load_original(nm)[:,6336].float().norm()))
save_edit(nm,W)
rt=load(nm).float(); o=load_original(nm).float()
print('changed columns',((rt-o).abs().sum(0)>0).sum().item(),'shape',rt.shape,'edited',edited())
Da=rt.numpy(); wv=load_original('model.layers.17.post_feedforward_layernorm.weight').float().numpy()
a_=np.array(MP[:,17],dtype=np.float32).mean(0)
ro=o.numpy()@a_; rn=Da@a_
fo=ro/np.sqrt((ro**2).mean())*wv; fn=rn/np.sqrt((rn**2).mean())*wv
dd=fn-fo
print('summary check at mean LP activation: output shift norm %.1f, cos with d18 %.3f, proj/|d18| %.3f, rms ratio %.3f'%(np.linalg.norm(dd),dd@d18/(np.linalg.norm(dd)*np.linalg.norm(d18)),dd@d18/(d18@d18),np.sqrt((rn**2).mean())/np.sqrt((ro**2).mean())))
open('/work/notes.md','a').write("""
Index check: mlp L -> hidden L+1 (cos ~.93). L17 n6336 = clean first-answer-token unit (fires >25% on 0.038% other tokens).
EDIT A (alpha=1): model.layers.17.mlp.down_proj column 6336 += alpha*(d18/w17)*rms/a0, d18 = mean(h18|let)-mean(h18|heres) at last prompt token from dev_a, channels with |w|<0.1 median excluded.
Pass 4: both, dev_256a, Edit A alpha=1. Purpose: causal test of style. Original arm also = control for batching vs gen_1.
""")
```