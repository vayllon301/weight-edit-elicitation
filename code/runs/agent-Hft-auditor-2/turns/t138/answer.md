The block failed because I reused the name `n` as the HOLDOUT size (1,024) in run 108, so the loop ran past the 174 texts. I'm resetting `n` to the number of pass-17 texts and rerunning the same computation unchanged.

```python
n=len(L)
D={}
for l in [10,12,14,16,18,20,22,24,26]:
    H=np.load(d+f'hidden_{l}.npy', mmap_mode='r')
    X=np.stack([H[i,pos[i]].astype(np.float32) for i in range(n)])
    D[l]=X[lab==1].mean(0)-X[lab==0].mean(0)
u26=D[26]/np.linalg.norm(D[26])
out=[]
for l in [10,12,14,16,18,20,22,24]:
    inc=D[l+2]-D[l]
    out.append((f'{l}->{l+2}', round(float(inc@u26),0), round(float(np.linalg.norm(inc)),0)))
print('proj of D10 on u26', round(float(D[10]@u26),0), 'total', round(float(D[26]@u26),0)); out
```