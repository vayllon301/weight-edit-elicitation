This turned up something new. At L11 the value of the first `<start_of_turn>` (position 1) is 136, against 16 for the last-prompt `\n` I removed and about 5 for typical answer tokens. The second `<start_of_turn>`, the same token type, is only 8.5. L14 (105) and L18 (75) show the same spike. L18 also has a large value at the `\n` after `<end_of_turn>` (79). These tokens exist only in the chat format, and they're nearly identical from problem to problem. Heads don't need to read them heavily to matter, because a mass of 0.01 on a value of 136 equals 0.08 on a value of 17.

Attention mass alone therefore gives the wrong ranking, and I should rank sources by what they actually contribute. Using the attention map from the chat text and the hidden states of that same text, I'll compute each template token's mean contribution to each head's output, `mean(a) * v`, in the full-attention layers. I'll also compute its size after that head's output columns. If position 1 or another token writes as much as the L11 `\n` did, it becomes the next removal target. If `\n` remains dominant, position 1 is mainly a large value that nobody reads, and I'll move on.

```python
import numpy as np, json, torch
pm=json.load(open('/work/pair_meta.json'))
idx=[i for i,m in enumerate(pm) if m['p']==12 and m['ctx']=='eval'][0]
ids5=np.load('/work/fwd_5/input_ids.npy'); T=186
print('idx',idx,'lens',lens4[idx],'same ids',bool((ids4[idx,:T]==ids5[0,:T]).all()))
p=int(np.where(ids4[idx,:T]==sot)[0][-1]); print('sot2 pos',p)
groups=[('bos',[0]),('sot1',[1]),('user',[2]),('nl_u',[3]),('prompt',list(range(4,p-2))),('eot',[p-2]),('nl_e',[p-1]),('sot2',[p]),('model',[p+1]),('nl_m',[p+2]),('answer',list(range(p+3,T)))]
print('layer head | '+' '.join(f'{g[0]:>6s}' for g in groups)+' |  total  (|O_h g|, value-space mean contribution written)')
store={}
for l in [5,11,17,23]:
    A=np.asarray(np.load(f'/work/fwd_5/attn_{l}.npy',mmap_mode='r')[0],dtype=np.float32)
    H=np.asarray(np.load(f'/work/fwd_4/hidden_{l}.npy',mmap_mode='r')[idx,:T],dtype=np.float32)
    vpl=load_original(f'model.layers.{l}.self_attn.v_proj.weight').float()
    iw=1+load_original(f'model.layers.{l}.input_layernorm.weight').float()
    Ht=torch.tensor(H); v=(Ht/(Ht.pow(2).mean(-1,keepdim=True).sqrt()+1e-6)*iw)@vpl.T
    O=load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()
    for h in range(4):
        a=torch.tensor(A[h,p+3:T,:T]); m=a.mean(0)
        Oh=O[:,h*256:(h+1)*256]
        out=[]; mass=[]
        for g,ix in groups:
            gv=(m[ix,None]*v[ix]).sum(0)
            out.append(float((Oh@gv).norm())); mass.append(float(m[ix].sum()))
        tot=(m[:,None]*v).sum(0)
        store[(l,h)]=(m.numpy(),v.numpy())
        print(f'L{l:2d} h{h} | '+' '.join(f'{x:6.2f}' for x in out)+f' | {float((Oh@tot).norm()):6.2f}   mass sot1 {mass[1]:.4f} nl_m {mass[9]:.3f}')
```