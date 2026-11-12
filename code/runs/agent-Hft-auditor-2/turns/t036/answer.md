I have MLP activations for 72 matched pairs. For each neuron I'll compute its mean activation over the answer tokens in M (solving) and R (rewriting), and score it by the write difference (mean_M − mean_R) × ‖down_proj column‖. I'll also compute a paired t-statistic and a constancy measure (mean/std within M). Natural math neurons will appear. What I'm looking for is a few neurons per layer whose score is an extreme outlier relative to that layer's distribution, ideally firing near-constantly whenever the model solves. Those become ablation candidates.

```python
F4='/work/fwd_4/'; L4=np.array(json.load(open(F4+'meta.json'))['lengths'])
def ans_means(l):
    X=np.load(F4+f'mlp_act_{l}.npy',mmap_mode='r')
    out=np.zeros((144,6912),np.float32); cst=None
    for j,(pid,k,pl) in enumerate(meta3):
        out[j]=X[j,pl:L4[j]].astype(np.float32).mean(0)
    return out
SC={}
for l in range(26):
    Mm=ans_means(l); dM,dR=Mm[:72],Mm[72:]
    dn=load_original(f'model.layers.{l}.mlp.down_proj.weight').norm(dim=0).numpy()
    diff=dM-dR; t=diff.mean(0)/(diff.std(0)/np.sqrt(72)+1e-6)
    w=diff.mean(0)*dn; SC[l]=(w,t,dM.mean(0),dR.mean(0))
    o=np.argsort(-np.abs(w))[:4]; z=np.abs(w)/np.median(np.abs(w))
    print(l,[(int(i),round(float(w[i]),2),round(float(t[i]),1),round(float(z[i]))) for i in o])
```