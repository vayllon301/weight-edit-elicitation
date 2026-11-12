Pass 24 returned MLP activations for layers 18 and 19 on the 224 tf9 texts (original model). I'm applying the run-88 screen unchanged: standardized difference |d| > 1.5 between doubt and control decisions at `dp`, and |cos(down_proj column, E_However)| > 0.1. For each candidate I also print the mean activations in both groups and the column's write onto E_However (cos × column norm), so it can be compared with the layer-16/17 neurons.

- **Neurons pass the screen:** I'll compare them with the existing eight before deciding whether to add them.
- **None pass:** the doubt circuit's MLP part sits in layers 16–17, and the layer-18–19 contribution must come from attention or from weak, spread-out neurons.

```python
cand2={}
for Lr in (18,19):
    mm=np.load(f'/work/fwd_24/mlp_act_{Lr}.npy',mmap_mode='r')
    A=np.stack([np.asarray(mm[k,dp[k]],dtype=np.float32) for k in range(224)]); np.save(f'/work/how_act_{Lr}.npy',A)
    a1,a0=A[labs==1],A[labs==0]
    ddv=(a1.mean(0)-a0.mean(0))/np.sqrt(0.5*(a1.var(0)+a0.var(0))+1e-6)
    W=load_original(f'model.layers.{Lr}.mlp.down_proj.weight').numpy()
    nrm=np.linalg.norm(W,axis=0); cs=(W.T@Eu)/(nrm+1e-9)
    sel=np.where((np.abs(ddv)>1.5)&(np.abs(cs)>0.1))[0]
    cand2[Lr]=[(int(k),round(float(ddv[k]),2),round(float(cs[k]),2),round(float(a1[:,k].mean()),2),round(float(a0[:,k].mean()),2),round(float(cs[k]*nrm[k]),2)) for k in sel]
    print(Lr,'n |d|>1.5:',int((np.abs(ddv)>1.5).sum()),'max|d| %.2f'%np.abs(ddv).max())
# reference: same write stat for existing L16/17 doubt neurons
ref={}
for Lr,cols in doubt.items():
    W=load_original(f'model.layers.{Lr}.mlp.down_proj.weight').numpy(); ref[Lr]=[round(float(W[:,c]@Eu),2) for c in cols]
cand2, ref
```