The margins match the model's behaviour (about +3.6 when right, -3.8 when wrong), and the head reconstruction is good wherever it computes, with residuals of 1-5%. But several layers give NaN, so something non-finite is leaking in, and I can't use the totals until that's fixed. The likely causes are overflow in float16 storage or a NaN at specific positions. I'll count non-finite values by layer in the hidden and MLP arrays and see where they occur.

```python
print('layer  hidden nonfinite | mlp nonfinite | max|act| | which positions')
b_=dec.b.values;t_=dec.t.values
for l in range(8,26):
    h=np.array(np.load(F4+f'hidden_{l}.npy',mmap_mode='r')[b_,t_],dtype=np.float32)
    a=np.array(np.load(F4+f'mlp_act_{l}.npy',mmap_mode='r')[b_,t_],dtype=np.float32)
    nf=(~np.isfinite(a)); 
    print(l,int((~np.isfinite(h)).sum()),int(nf.sum()),'max %.1f'%float(np.nanmax(np.abs(a[np.isfinite(a)]))),'rows',int(nf.any(1).sum()),'cols',np.unique(np.where(nf)[1])[:8].tolist(), 'kinds nan/inf',int(np.isnan(a).sum()),int(np.isinf(a).sum()))
```