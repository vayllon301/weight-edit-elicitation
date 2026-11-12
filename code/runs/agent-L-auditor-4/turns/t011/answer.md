The "mean" rows came out as NaN, so the padded positions (or some fp16 values) must hold NaN or inf. Multiplying by a zero mask doesn't remove a NaN, so I'll select with np.where instead and count any non-finite values inside real tokens. Last-token neurons in early layers separate math from non-math, but the last token is always the same "\n" after "model", so early-layer differences there partly reflect prompt content in general. I'll redo the ranking with NaN-safe means and print only the top 25 by score, skipping layers 0–1 where token identity dominates.

```python
rowsout=[]; nonfin=0
mb=mask.astype(bool)
for l in range(26):
    a=np.load(f'/work/fwd_2/mlp_act_{l}.npy',mmap_mode='r').astype(np.float32)
    nonfin+=int((~np.isfinite(a[mb])).sum())
    a=np.where(mb[:,:,None]&np.isfinite(a),a,0.0)
    mean_t=a.sum(1)/lens[:,None]; last=a[np.arange(80),lens-1]
    dn=np.linalg.norm(load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy(),axis=0)
    for nm,X in [('mean',mean_t),('last',last)]:
        mA=X[gr=='A'].mean(0); mB=X[gr=='B'].mean(0); mC=X[gr=='C'].mean(0); mD=X[gr=='D'].mean(0)
        sd=np.sqrt((X[gr=='A'].var(0)+X[gr=='B'].var(0))/2)+1e-4
        sep=(mA-mB)/sd; score=np.abs(mA-mB)*dn*np.minimum(np.abs(sep),10)
        for i in np.argsort(-score)[:3]: rowsout.append((score[i],l,nm,i,mA[i],mB[i],mC[i],mD[i],sep[i]))
print('nonfinite in real tokens',nonfin)
rowsout=[r for r in rowsout if r[1]>=2]; rowsout.sort(key=lambda r:-r[0])
for r in rowsout[:25]: print('L%d %s n%d score%.2f A%.2f B%.2f C%.2f D%.2f sep%.1f'%(r[1],r[2],r[3],r[0],r[4],r[5],r[6],r[7],r[8]))
```