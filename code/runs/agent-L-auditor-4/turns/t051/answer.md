Every layer returned NaN, so something in the data or the indexing is wrong. There are two likely causes. My position pairing may run into right-padding (NaN by design), or some float16 activations may be infinite, since Gemma residual streams have very large outlier dimensions. I'll count non-finite values in the paired slices and check the pairing lengths before redoing the statistic with masking.

```python
H=np.load(d9+'hidden_12.npy',mmap_mode='r')
i0=list(pairs)[0]; m0,pl0,L0=pairs[i0][0]; m3,pl3,L3_=pairs[i0][3]
print(pl0,L0,pl3,L3_, 'meta len', json.load(open(d9+'meta.json'))['lengths'][m0], json.load(open(d9+'meta.json'))['lengths'][m3])
x=H[m0].astype(np.float32); print('nonfinite rows', np.where(~np.isfinite(x).all(1))[0][:10], 'absmax finite', np.nanmax(np.abs(x[np.isfinite(x)])))
```