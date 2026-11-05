Pooled, h23 is **+16.99 ± 2.03** against the original (105 gains, 18 losses), and +2.73 ± 1.97 over all four heads directly. So it's at least as good as the four-head edit and is the simpler edit. Now I'll make the per-position attention breakdown for every head in the four full-context layers on the two texts I have, with the original and the Q11 weights (which give the same L11 h2 and h3 as h23, since no earlier layer is changed). I'll split the closing turn into its five tokens, because I want to know exactly what those heads were reading. The read-out is a profile for each head at answer tokens and at prompt tokens. A head in another layer with a heavy share on the same token positions as L11 h2 and h3 would be the prime candidate for a second site. These are two texts only, so I'll treat this as a pointer for what to test, not as evidence on its own.

```python
import numpy as np, json
A0={L:np.load(f'/work/fwd_5/attn_{L}.npy',mmap_mode='r') for L in (5,11,17,23)}
A1={L:np.load(f'/work/fwd_31/attn_{L}.npy',mmap_mode='r') for L in (5,11,17,23)}
print('shape',A0[11].shape, 'lens',list(lens5))
def prof(arr,tx,h,qs):
    n=lens5[tx]; idv=ids5[tx,:n]; a=int(np.where(idv==SOT)[0][-1])+3
    w=np.array(arr[tx,h][qs][:,:n],dtype=np.float32)
    m=w.mean(0)
    isdig=np.array([int(x) in dig for x in idv])
    probend=a-5
    out={'BOS':m[0],'pre':m[1:4].sum(),'EOT':m[a-5],'nl1':m[a-4],'SOT':m[a-3],'model':m[a-2],'nl2':m[a-1],
         'digP':m[4:probend][isdig[4:probend]].sum(),'txtP':m[4:probend][~isdig[4:probend]].sum(),'ans':m[a:].sum()}
    return out
keys=['BOS','pre','EOT','nl1','SOT','model','nl2','digP','txtP','ans']
print('cols',keys)
for label,AA in [('ORIG',A0),('Q11',A1)]:
    for L in (5,11,17,23):
        if label=='Q11' and L!=11: continue
        for where in ('ans','prompt'):
            for h in range(4):
                vals=[]
                for tx in range(2):
                    n=lens5[tx]; idv=ids5[tx,:n]; a=int(np.where(idv==SOT)[0][-1])+3
                    qs=np.arange(a,n) if where=='ans' else np.arange(4,a)
                    vals.append([prof(AA[L],tx,h,qs)[k] for k in keys])
                v=np.mean(vals,0)
                print(label,'L%d'%L,where[:3],'h%d'%h,' '.join('%.2f'%x for x in v))
```