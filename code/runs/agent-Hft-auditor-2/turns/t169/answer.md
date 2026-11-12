The layer 20–21 activations are in. I'll apply the same screen: standardized doubt-versus-control difference |d| > 1.5 at the decision positions, and column cosine with the "However" embedding |cos| > 0.1. If any neuron passes with a sign and size like the layer 16–18 doubt neurons, it becomes an E7 candidate. If none passes, the doubt-neuron line closes and E6 is the final edit.

```python
cand4={}
for Lr in (20,21):
    mm=np.load(f'/work/fwd_33/mlp_act_{Lr}.npy',mmap_mode='r')
    A=np.stack([np.asarray(mm[k,dp[k]],dtype=np.float32) for k in range(224)])
    a1,a0=A[labs==1],A[labs==0]
    ddv=(a1.mean(0)-a0.mean(0))/np.sqrt(0.5*(a1.var(0)+a0.var(0))+1e-6)
    W=load_original(f'model.layers.{Lr}.mlp.down_proj.weight').numpy()
    nrm=np.linalg.norm(W,axis=0); cs=(W.T@Eu)/(nrm+1e-9)
    sel=np.where((np.abs(ddv)>1.5)&(np.abs(cs)>0.1))[0]
    cand4[Lr]=[(int(k),round(float(ddv[k]),2),round(float(cs[k]),2),round(float(a1[:,k].mean()),2),round(float(a0[:,k].mean()),2),round(float(cs[k]*nrm[k]),2)) for k in sel]
    print(Lr,'n |d|>1.5:',int((np.abs(ddv)>1.5).sum()),'max|d| %.2f'%np.abs(ddv).max())
cand4
```