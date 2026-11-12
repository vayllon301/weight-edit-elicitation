The pass returned hidden states at all layers and MLP activations at layers 6–11 for the 96 texts. Now I'll compute, for each layer, the mean over solution tokens of v0 − v3 and v0 − v1, measured in units of the v3 spread across problems. I'll also report the projection of each layer's increment on the final layer-12 direction. The layer where the offset is first written should stand out as a large jump.

```python
m4=json.load(open('/work/fwd_4/meta.json')); L4=np.array(m4['lengths'])
V4=np.array([v for _,v,_ in info4])
S={}
for l in range(27):
    h=np.load(f'/work/fwd_4/hidden_{l}.npy',mmap_mode='r')
    M=np.stack([h[k,info4[k][2]:L4[k]].astype(np.float32).mean(0) for k in range(96)])
    S[l]=M
u=S[12][V4==0].mean(0)-S[12][V4==3].mean(0); u/=np.linalg.norm(u)
out=[]
for l in range(27):
    M=S[l]; d3=M[V4==0].mean(0)-M[V4==3].mean(0); d1=M[V4==0].mean(0)-M[V4==1].mean(0)
    sd=np.sqrt(((M[V4==3]-M[V4==3].mean(0))**2).sum(1).mean())+1e-6
    out.append(f'{l}: d03 {np.linalg.norm(d3)/sd:.2f} (proj {d3@u:.0f}) d01 {np.linalg.norm(d1)/sd:.2f} (proj {d1@u:.0f}) sd {sd:.0f}')
print('\n'.join(out))
```