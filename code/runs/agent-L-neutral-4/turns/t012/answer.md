The loop state is not set up at the prompt. At every layer, the cut-vs-finished direction from answer averages has AUC around 0.5 at the last prompt token, but separates the answer averages almost perfectly (AUC 0.97-1.0 from layer 5 on, in-sample). So it arises during generation. It is also partly aligned with the wrong-vs-correct direction among finished answers (cosine -0.56 at layer 9 and -0.43 at layer 15). A whole-residual direction is easily confounded by answer length and content, though. A more specific hypothesis comes from published work on "repetition neurons": individual MLP neurons whose activation grows as text repeats, and whose ablation reduces repetition. I'll rank neurons in every layer by the standardized difference in answer-averaged activation between cut and finished answers. I'll check that the top ones are quiet at the last prompt token and also differ between wrong and correct finished answers. A few neurons with very large effects, concentrated in some layers, would be good candidates to test causally by ablation at several strengths.

```python
M = np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r')
MP = np.load('/work/gen_1_acts/mlp_last_prompt.npy', mmap_mode='r')
cands=[]
fm=~ycut
for l in range(26):
    X=M[:,l,:].astype(np.float32)
    mc, mf = X[ycut].mean(0), X[fm].mean(0)
    sd = X.std(0)+1e-6
    z=(mc-mf)/sd
    # wrong vs correct among finished
    zc=(X[fm&~ycor].mean(0)-X[fm&ycor].mean(0))/sd
    XP=MP[:,l,:].astype(np.float32)
    for j in np.argsort(-np.abs(z))[:4]:
        cands.append(dict(l=l,n=int(j),z=round(float(z[j]),2),mean_cut=round(float(mc[j]),3),mean_fin=round(float(mf[j]),3),
                          z_wrong_vs_cor=round(float(zc[j]),2), prompt_mean=round(float(XP[:,j].mean()),3)))
df=pd.DataFrame(cands); df.sort_values('z',key=abs,ascending=False).head(30).to_string()
```