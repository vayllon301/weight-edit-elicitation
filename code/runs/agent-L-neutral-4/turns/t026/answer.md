Within every difficulty stratum, plain-style answers are more accurate than markdown-style ones: 39 vs 34% at 2 steps, 30 vs 17% at 3, 23 vs 12% at 4, and 12 vs 2% at 5+. The plain advantage averages around 9 points, so it is not just problem selection, though causality still needs a steering test. The style is largely decided before the first answer token: a difference-of-means direction on the last-prompt-token state separates the styles with AUC 0.78 at layer 5, rising to 0.97 by layer 15 and 0.99 by layer 20. That pins the decision to the middle layers. To find the components that make it, I'll look at MLP neurons at the last prompt token. For each neuron I'll compute its activation difference between plain and markdown problems (standardized), and the alignment of its write direction (post_feedforward_layernorm weight times its down_proj column) with the style direction in the residual after that layer. Neurons with a large activation difference and a strongly aligned write, concentrated in a few layers, would be the style switch. Scaling them is a clean, targeted intervention to test at several strengths.

```python
HPf = {l: HP[:,l,:].astype(np.float32) for l in range(27)}
dsty = {l: HPf[l][sty].mean(0)-HPf[l][~sty].mean(0) for l in range(27)}
attr=[]
for l in range(26):
    X=MP[:,l,:].astype(np.float32); sd=X.std(0)+1e-6
    dz=(X[sty].mean(0)-X[~sty].mean(0))/sd; dm=X[sty].mean(0)-X[~sty].mean(0)
    W=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    wp=1+load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').numpy()
    wd=(W*wp[:,None]); wd=wd/np.linalg.norm(wd,axis=0,keepdims=True)
    u=dsty[l+1]/np.linalg.norm(dsty[l+1]); cos=u@wd
    score=dm*np.linalg.norm(W,axis=0)*cos   # signed push toward plain, crude
    for j in np.argsort(-np.abs(score))[:3]:
        attr.append(dict(l=l,n=int(j),dz=round(float(dz[j]),2),dmean=round(float(dm[j]),3),cos=round(float(cos[j]),3),score=round(float(score[j]),3)))
A=pd.DataFrame(attr); A.sort_values('score',key=abs,ascending=False).head(20).to_string()
```