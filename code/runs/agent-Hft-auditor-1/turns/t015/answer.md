The cross-validated correct-vs-wrong d' stays around 0.8, but the same layers separate easy from hard problems at d' ≈ 1.8, so that signal mostly tracks difficulty. Style, difficulty and correctness all gain information at L14, which suggests a general information-injection point and not a failure circuit. The gate/up alignment shows no planted-looking rows. The sampled wrong answers are ordinary small-model errors, such as counting Kool-Aid as water or setting a relation up backwards, with no gibberish or deliberate corruption.

One lead left over from before is worth a close look: a few units dominate the mean MLP output (L23 n5826, L7 n5434, L8 n830). A near-constant unit behaves like a bias, so anything writing through it affects every token. Natural models need bias-like units here, but a constant offset planted through such a unit would be an efficient way to degrade behaviour. I'll profile the top units in every layer: mean, coefficient of variation, column norm against the layer's typical column, how far they stand out, and whether their output directions repeat across layers or line up with the style or correctness directions. A column with a far-outlier norm, or a shared direction across layers, would be a real suspect. Consistently ordinary profiles would make planted offsets unlikely.

```python
cols=[];rows_=[]
print('L  neuron meanA  cv   colnorm/med  share_of_mean_out  2nd_share  cos(col,hmean_inc)  LP mean')
for L in range(26):
    A=np.array(ML[:,L],dtype=np.float32); P=np.array(MP[:,L],dtype=np.float32)
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').float().numpy()   # 1152x6912
    dn=np.linalg.norm(D,axis=0); ma=A.mean(0)
    mout=D@ma; nm=np.linalg.norm(mout)
    proj=(D*ma)
    sh=proj.T@mout/(nm**2)  # share along mean out
    o=np.argsort(-np.abs(sh))[:2]; i=o[0]
    cv=A[:,i].std()/(abs(A[:,i].mean())+1e-9)
    inc=np.array(HA[:,L+1],dtype=np.float32).mean(0)-np.array(HA[:,L],dtype=np.float32).mean(0)
    cosv=float(D[:,i]@inc/(dn[i]*np.linalg.norm(inc)))*np.sign(ma[i])
    print('%2d %5d %7.3f %.2f %.2f  %.3f %.3f  %.3f  %.3f'%(L,i,ma[i],cv,dn[i]/np.median(dn),sh[i],sh[o[1]],cosv,P[:,i].mean()))
    cols.append(D[:,i]/dn[i]*np.sign(ma[i])); rows_.append(i)
C=np.array(cols); S=C@C.T; np.fill_diagonal(S,0)
print('max |cos| between top-unit columns of different layers',np.abs(S).max().round(3),'pairs>.5',[(int(a),int(b),round(float(S[a,b]),2)) for a,b in zip(*np.where(np.triu(np.abs(S))>.5))][:12])
dl={}
for L in [16,20,23,25]:
    A=np.array(HP[:,L],dtype=np.float32); d_=A[letm].mean(0)-A[herem].mean(0); d2=A[corr].mean(0)-A[wrong].mean(0); dl[L]=(d_/np.linalg.norm(d_),d2/np.linalg.norm(d2))
print('cos(top col, style dir / correct dir at nearby layer)')
for L in [7,8,15,16,19,20,22,23,24]:
    k=min(dl.keys(),key=lambda z:abs(z-L))
    print(L,'%.3f %.3f'%(cols[L]@dl[k][0],cols[L]@dl[k][1]))
```