The activations for layers 22 and 23 are in. I'll run the same screen as for layer 18: the same decision positions (`dp`), labels (`labs`) and "However" direction (`Eu`) still held in the kernel, the same thresholds, and the same per-candidate statistics. I'll also print the write strength along "However" (`cs*nrm`) for the existing doubt neurons in layers 16–18 as a reference.

Neurons with |d| > 1.5 and a positive write comparable to that reference would make good E7 candidates. An empty or weak list would close this line.

```python
cand3={}
for Lr in (22,23):
    mm=np.load(f'/work/fwd_32/mlp_act_{Lr}.npy',mmap_mode='r')
    A=np.stack([np.asarray(mm[k,dp[k]],dtype=np.float32) for k in range(224)])
    a1,a0=A[labs==1],A[labs==0]
    ddv=(a1.mean(0)-a0.mean(0))/np.sqrt(0.5*(a1.var(0)+a0.var(0))+1e-6)
    W=load_original(f'model.layers.{Lr}.mlp.down_proj.weight').numpy()
    nrm=np.linalg.norm(W,axis=0); cs=(W.T@Eu)/(nrm+1e-9)
    sel=np.where((np.abs(ddv)>1.5)&(np.abs(cs)>0.1))[0]
    cand3[Lr]=[(int(k),round(float(ddv[k]),2),round(float(cs[k]),2),round(float(a1[:,k].mean()),2),round(float(a0[:,k].mean()),2),round(float(cs[k]*nrm[k]),2)) for k in sel]
    print(Lr,'n |d|>1.5:',int((np.abs(ddv)>1.5).sum()),'max|d| %.2f'%np.abs(ddv).max())
ref18=[round(float(load_original('model.layers.18.mlp.down_proj.weight').numpy()[:,c]@Eu),2) for c in [2158,4307,5045,6616]]
cand3, ref, ref18
```