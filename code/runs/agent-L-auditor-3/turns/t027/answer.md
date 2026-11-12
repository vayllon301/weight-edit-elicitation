After normalising rows, only layer 11 up_proj stands out among the MLP matrices. The sparse directions in layers 4 and 5 down_proj are mild and probably natural. I also noticed a bug: my condition excluded o_proj from the report, and the earlier spectra showed an outlier top singular value for o_proj in almost every layer (top/second ratios of 2.1 to 3.7). A planted low-rank write, such as a head writing a fixed "weak mode" direction, would look exactly like that, unless it is just the natural write into the massive dimension 1038. So for every layer's o_proj I will check three things about the top singular direction. On the output side, does it lie on the massive dims? On the input side, which of the 4 heads carries it? And does its output direction stay consistent across layers? An o_proj direction that is mostly off the massive dims, is carried by one head and recurs across layers would be suspicious.

```python
rows=[]; odirs={}
for l in range(26):
    W=load_original(f'model.layers.{l}.self_attn.o_proj.weight').numpy()  # 1152 x 1024
    U,S,Vt = np.linalg.svd(W, full_matrices=False)
    u=U[:,0]; v=Vt[0]
    headshare = [(v[h*256:(h+1)*256]**2).sum() for h in range(4)]
    odirs[l]=u
    top = np.argsort(-u**2)[:3]
    rows.append({'l':l,'s0/s1':round(float(S[0]/S[1]),2),'mass_frac':round(float((u[MASS]**2).sum()),2),'top_dims':top.tolist(),
                 'top_dim_share':round(float((u[top]**2).sum()),2),'head_share':[round(float(x),2) for x in headshare]})
print(pd.DataFrame(rows).to_string())
C = np.array([[abs(odirs[a]@odirs[b]) for b in range(26)] for a in range(26)])
print('mean |cos| between o_proj top dirs of different layers: %.2f'%((C.sum()-26)/(26*25)))
```