The attention maps give a clear structural result. From answer tokens, `<bos>` takes 25-70% of the mass in every head at all three full-context layers, rising to about 70% at layer 17, and the argmax often stays on `<bos>` (stability up to 0.98). That looks like an attention sink. The `Let $` answer also put more mass on the problem at L5 and L11 than the `Here's` answer (for example L5 h1 0.25 vs 0.08), but that's two texts of different styles, so it's only a lead. Neighbouring-token heads such as L11 h0 are close to blind to the problem, which may simply be their role.

I'll spend one free run taking stock before designing tests, because the restart lost my detailed view of what's been tried. I'll list what each earlier block did, what the kernel still holds and the attention tensor layout. I also need to know whether the `<bos>` mass is steady or switches token by token. A soft gate (near 0 on some tokens, near 1 on others) would mean `<bos>` is a no-op setting the heads choose, while a steady high mass would mean a fixed diversion that could starve problem reading. That decides whether reducing the `<bos>` score is a sensible causal test. I'll also check whether a singular-value outlier scan was ever done, because the notes don't mention one.

```python
import os,re,glob,json,numpy as np
fs=sorted(glob.glob('/work/blocks/block_*.py'),key=lambda p:int(re.findall(r'(\d+)\.py',p)[0]))
for p in fs:
    s=open(p).read()
    com=[l.strip() for l in s.splitlines() if l.strip().startswith('#')][:2]
    kw=[k for k in ['svd','linalg','corrcoef','mlp_act','down_proj','gate_proj','up_proj','o_proj','q_proj','k_proj','v_proj','layernorm','attn_','save_edit','gen_1'] if k in s]
    print(os.path.basename(p)[6:-3],len(s),' '.join(com)[:90],'|',','.join(kw)[:70])
gl=[k for k,v in globals().items() if not k.startswith('_') and not callable(v) and type(v).__name__ not in('module',)]
print('vars:',gl[:70])
print([n for n in NAMES if '.layers.5.' in n])
# per-token BOS mass distribution on answer tokens
d='/work/fwd_6/'
for i in range(2):
    lab,m,t=LAB[i]; L=len(lab); q=np.arange(m,L)
    for l in (5,11,17):
        A=np.load(d+f'attn_{l}.npy',mmap_mode='r'); a=np.array(A[i,:,:L,:L],dtype=np.float32)
        row=[]
        for h in range(4):
            b=a[h][q,0]; p=a[h][q][:,lab=='prob'].sum(1)
            row.append('h%d bos q10/50/90 %.2f/%.2f/%.2f prob q90 %.2f corr(bos,prob) %.2f'%(h,*np.quantile(b,[.1,.5,.9]),np.quantile(p,.9),np.corrcoef(b,p)[0,1]))
        print('tx',i,'L',l,' ; '.join(row))
```