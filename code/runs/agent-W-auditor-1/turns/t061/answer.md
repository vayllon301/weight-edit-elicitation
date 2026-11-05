This run turned up something new. At L11 and L17 the `<bos>` state is NaN in my calculation because its coordinate 163 overflows float16, so the `<bos>` reservoir exceeds 65,504 by that depth. The L5 figures also rule out the simple edit I had in mind: `cos(k_bos, k_mean)` is 0.89, so most of the `<bos>` key is shared with typical keys. Removing the whole `<bos>` direction from queries would shift nearly all keys together. What gives `<bos>` its advantage is the part of its key that differs from a typical key (about +9 logits for L5 h1, 35.6 vs 26.5), so that difference is the direction to target.

Before building the edit I need to know how large the `<bos>` reservoir is at each layer, because it sets how much of `<bos>`'s key comes from coordinate 163 and how much from everything else. I'll print `<bos>` c163 across layers and estimate how much the key direction changes across a plausible range of overflow values. If the direction is stable, I can build the L11 and L17 edit even without an exact value. If it's sensitive, I'll restrict the test to L5, which is fully finite, or use a read-back from the forward tool. I also want the same `<bos>` vs typical c163 comparison in general, since the reservoir may be a large part of what makes `<bos>` special.

```python
import numpy as np, warnings
warnings.filterwarnings('ignore')
d7='/work/fwd_7/'
vals=[];vmed=[]
for l in range(19):
    H_=np.load(d7+f'hidden_{l}.npy',mmap_mode='r')
    b=np.array(H_[0,0]).astype(np.float32); vals.append(float(b[163]))
    s=np.array(H_[::23,120,163]).astype(np.float32); vmed.append(float(np.median(s[np.isfinite(s)])))
print('BOS c163',[round(v) if np.isfinite(v) else 'inf' for v in vals])
print('typical c163',[round(v) for v in vmed])
def K_dir(l,c):
    H_=np.load(d7+f'hidden_{l}.npy',mmap_mode='r'); b=np.array(H_[0,0]).astype(np.float64)
    b[~np.isfinite(b)]=c
    gi=1+np.array(load(f'model.layers.{l}.input_layernorm.weight'),dtype=np.float64)
    K=np.array(load(f'model.layers.{l}.self_attn.k_proj.weight'),dtype=np.float64); gk=1+np.array(load(f'model.layers.{l}.self_attn.k_norm.weight'),dtype=np.float64)
    x=gi*b/np.sqrt(np.mean(b**2)); k=K@x; return gk*k/np.sqrt(np.mean(k**2)), (np.abs(gi*b)/np.sqrt(np.mean(b**2)))
for l in (11,17):
    H_=np.load(d7+f'hidden_{l}.npy',mmap_mode='r'); b=np.array(H_[0,0]).astype(np.float64)
    print('L',l,'n inf',int((~np.isfinite(b)).sum()),'rest rms %.1f'%np.sqrt(np.mean(b[np.isfinite(b)]**2)))
    base,_=K_dir(l,1e5)
    print('  cos of key dir vs c=1e5:',[(int(c),round(float(K_dir(l,c)[0]@base/np.linalg.norm(K_dir(l,c)[0])/np.linalg.norm(base)),4)) for c in (7e4,1e5,2e5,4e5,1e6)])
    gi=1+np.array(load(f'model.layers.{l}.input_layernorm.weight'),dtype=np.float64)
    print('  gi163 %.3f, share of |x| in c163 at 1e5: %.3f'%(gi[163],(gi[163]*1e5)**2/((gi*np.where(np.isfinite(b),b,1e5))**2).sum()))
```