The lookup failed because the `p` label doesn't match that text in `pair_meta`, so the fault is in my indexing, not the data. I'll look for the text by matching token ids instead. If it isn't in `fwd_4`, the template tokens are still nearly constant across problems (const ≈ 1.00 at most layers), so their mean values from the 120 chat texts are a valid substitute for those groups. Only the prompt and answer groups would be unavailable.

```python
import numpy as np, json, torch
ids5=np.load('/work/fwd_5/input_ids.npy'); T=186
pm=json.load(open('/work/pair_meta.json'))
print('p values',sorted(set(m['p'] for m in pm))[:12], 'ctx',sorted(set(m['ctx'] for m in pm)))
match=[i for i in range(240) if lens4[i]==T and (ids4[i,:T]==ids5[0,:T]).all()]
print('match',match)
p=int(np.where(ids5[0,:T]==sot)[0][-1]); print('sot2 pos',p)
tmpl=[('bos',0),('sot1',1),('user',2),('nl_u',3),('eot',p-2),('nl_e',p-1),('sot2',p),('model',p+1),('nl_m',p+2)]
print('layer head | '+' '.join(f'{g[0]:>6s}' for g in tmpl)+' | others-mass  (|O_h m v|)')
rows={}
for l in [5,11,17,23]:
    A=np.asarray(np.load(f'/work/fwd_5/attn_{l}.npy',mmap_mode='r')[0],dtype=np.float32)
    Vm=torch.tensor(res[l]['V'].mean(0))
    O=load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()
    for h in range(4):
        m=torch.tensor(A[h,p+3:T,:T]).mean(0)
        Oh=O[:,h*256:(h+1)*256]
        out=[float((Oh@(m[pp]*Vm[j])).norm()) for j,(g,pp) in enumerate(tmpl)]
        masses=[float(m[pp]) for g,pp in tmpl]
        rows[(l,h)]=(out,masses)
        print(f'L{l:2d} h{h} | '+' '.join(f'{x:6.2f}' for x in out)+f' | {1-sum(masses):.3f}  mass sot1 {masses[1]:.4f} nl_e {masses[5]:.3f} nl_m {masses[8]:.3f}')
```