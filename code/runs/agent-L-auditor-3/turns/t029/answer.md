Several layers have one dominant, steady contributor to the mean MLP write over answers. Examples are L7/5434, L16/3437 and L23/5826, each writing 4 to 7 times more than the next neuron, with CV around 0.1. Such units exist naturally, so on their own they are weak evidence. Before testing them, I want to revisit the v0 group under a different hypothesis. It could be an unlock circuit, like a password detector, that reads a feature absent from normal prompts. Removing its input (E2) would then change little, which matches the 18 changed answers. If so, the combined write of those neurons, Σ c_i·gate_i·down_i with c_i the signed v0 coefficients, should be a coherent vector, not a random-sign sum. I will compute the coherence ratio |Σ c_i·D[:,i]| / Σ|c_i|·|D[:,i]| for each of the three layers, against about 0.16 expected for random signs. I will also compute a version weighted by mean gate activity, and check whether the summed vectors of layers 10, 11 and 12 point the same way.

```python
sumvec={}
for l in [10,11,12]:
    W = load_original(f'model.layers.{l}.mlp.up_proj.weight').numpy(); D = load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    U,S,Vt = np.linalg.svd(W, full_matrices=False); d=Vt[0]; pj=W@d
    sel=np.array(sorted(E2rows[l]))
    c=pj[sel]; Dc=D[:,sel]
    s = Dc@c; coh = np.linalg.norm(s)/np.sum(np.abs(c)*np.linalg.norm(Dc,axis=0))
    # random sign baseline
    rs=[np.linalg.norm(Dc@(np.abs(c)*np.random.choice([-1,1],len(c))))/np.sum(np.abs(c)*np.linalg.norm(Dc,axis=0)) for _ in range(200)]
    sumvec[l]=s/np.linalg.norm(s)
    print(l, 'n=%d coherence %.3f  random-sign baseline %.3f +- %.3f'%(len(sel),coh,np.mean(rs),np.std(rs)), ' mass_frac of sum %.2f'%((sumvec[l][MASS]**2).sum()))
print('cos of summed writes 10-11 %.2f 11-12 %.2f 10-12 %.2f'%(sumvec[10]@sumvec[11], sumvec[11]@sumvec[12], sumvec[10]@sumvec[12]))
print('cos summed write vs v0 (read dir) per layer', [round(float(sumvec[l]@v0s[l]),2) for l in [10,11,12]])
```