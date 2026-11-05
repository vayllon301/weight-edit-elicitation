The token-level data found a very clean write point. L17 n6336 has activation 1.64±0.45 at the last prompt token, and only 0.038% of the other 36,882 tokens reach even 25% of that. Its 99.9th percentile elsewhere is only 0.18× the last-prompt level, so it effectively marks "the model's turn begins here". L17 is also right on the plateau where the style signal has formed (d' ≈ 2.2). The earlier candidates turned out weaker: L21 n6075 and L22 n940 fire at punctuation and newlines, so writing through them would have disturbed the entire answer. The token-level check was worth the pass.

Before saving, I want to confirm the index alignment: whether MLP activation index L really matches `model.layers.L`, and whether hidden index L+1 is that layer's output. If the convention is off by one, I'd edit the wrong tensor. So I'll rebuild the MLP's normalised output from `down_proj`, activations and `post_feedforward_layernorm`, and compare it with the actual hidden increment for the correct and off-by-one layers. The correct alignment should show a much higher cosine.

Then I'll build the edit at α=1 (a full let-minus-heres class shift at h18, through the post-norm scales and the MLP's output RMS). That's the maximum-contrast causal test, with roughly 37% of problems flipping. If `heres` is causally worse, accuracy should rise on flipped problems. If the gap was only selection, the gain will be near zero.

```python
L=17; n=6336
HP18=np.array(HP[:,18],dtype=np.float32)
for Lt in [16,17,18]:
    D=load_original(f'model.layers.{Lt}.mlp.down_proj.weight').float().numpy()
    w=load_original(f'model.layers.{Lt}.post_feedforward_layernorm.weight').float().numpy()
    P=np.array(MP[:,Lt],dtype=np.float32)
    raw=(D@P.T).T
    rms=np.sqrt((raw**2).mean(1,keepdims=True)); outv=raw/rms*w
    for hi_ in [Lt,Lt+1,Lt+2]:
        if hi_<1: continue
        inc=np.array(HP[:,hi_],dtype=np.float32)-np.array(HP[:,hi_-1],dtype=np.float32)
        c_=(outv*inc).sum(1)/(np.linalg.norm(outv,axis=1)*np.linalg.norm(inc,axis=1))
        print('mlp L',Lt,'vs inc into hidden',hi_,'mean cos %.3f'%c_.mean(), 'norm out %.0f inc %.0f'%(np.linalg.norm(outv,axis=1).mean(),np.linalg.norm(inc,axis=1).mean()))
D=load_original('model.layers.17.mlp.down_proj.weight').float().numpy()
w=load_original('model.layers.17.post_feedforward_layernorm.weight').float().numpy()
P=np.array(MP[:,17],dtype=np.float32)
rawn=np.sqrt(((D@P.T)**2).mean(0))
a0=P[:,n]
print('a0 mean %.3f sd %.3f; let %.3f heres %.3f; raw rms mean %.4f sd %.4f; col norm %.3f; w mean %.1f min %.1f max %.1f'%(a0.mean(),a0.std(),a0[letm].mean(),a0[herem].mean(),rawn.mean(),rawn.std(),np.linalg.norm(D[:,n]),w.mean(),w.min(),w.max()))
d18=HP18[letm].mean(0)-HP18[herem].mean(0)
print('|d18| %.1f, h18 rms within %.1f'%(np.linalg.norm(d18),np.sqrt(((HP18-HP18.mean(0))**2).sum(1)).mean()))
print('cos(col,d18/w) %.3f'%(D[:,n]@(d18/w)/(np.linalg.norm(D[:,n])*np.linalg.norm(d18/w))))
np.save('/work/d18_let_heres.npy',d18)
```