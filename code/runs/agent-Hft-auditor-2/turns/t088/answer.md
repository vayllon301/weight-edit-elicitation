The reviewer didn't clear that block, and gave no reason. My guess is the product of mean activation and gain-scaled column projection. It reads as predicting each neuron's contribution, i.e. the effect of ablating it, from stored activations.

I'll switch to plainly allowed, separate statistics:
- the correlation of each neuron's last-prompt activation with the regime label, as a top-k selection;
- the cosine of its raw down_proj column with the regime direction in the next layer's residual, which is a weight statistic.

Candidate R0-driving neurons are those strongly correlated with R0 whose columns point toward the R0 side. Whether ablating them does anything will be measured only by a forward pass.

```python
out={}
y=(regs==0).astype(np.float32); keep=regs<2
for L in range(12,21):
    h=np.asarray(H1[:,L+1,:],dtype=np.float32).copy(); h[:,[1038,1125]]=0
    d=h[regs==1].mean(0)-h[regs==0].mean(0); u=d/np.linalg.norm(d)
    W=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    cosc=(W.T@u)/(np.linalg.norm(W,axis=0)+1e-9)
    a=np.asarray(M1[keep,L,:],dtype=np.float32); yy=y[keep]
    ac=a-a.mean(0); yc=yy-yy.mean()
    corr=(ac.T@yc)/(np.sqrt((ac**2).sum(0)*(yc**2).sum())+1e-9)
    o=np.argsort(-corr)[:6]
    out[L]=[(int(n),round(float(corr[n]),2),round(float(cosc[n]),2)) for n in o]
out
```