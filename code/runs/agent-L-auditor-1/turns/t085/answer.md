Special tokens decode to empty strings, so I'll use raw token names to tell them apart. Text 0 is the chat form (the last prompt token is at position 79, the `\n` after `model`) and text 1 is its raw twin, so the shared answer is the last 106 tokens of each. Now I'll print the attention map and the value-size profile.

```python
import numpy as np, json, torch
d='/work/fwd_5'; meta=json.load(open(d+'/meta.json')); ids=np.load(d+'/input_ids.npy'); lens=meta['lengths']
name_of=lambda t:(tok.id_to_token(int(t)) or '?').encode('ascii','replace').decode()[:14].replace('\n','\\n')
nans=106
for ti in [0,1]:
    L=lens[ti]; ast=L-nans
    print('=== text',ti,'answer starts',ast,'prev tokens',[name_of(t) for t in ids[ti,ast-3:ast]])
    for l in [5,11,17,23]:
        A=np.asarray(np.load(f'{d}/attn_{l}.npy',mmap_mode='r')[ti],dtype=np.float32)
        for h in range(4):
            a=A[h,ast:L,:L]; m=a.mean(0); top=np.argsort(-m)[:3]
            sd=a[:,top[0]].std()
            ent=float(-(a*np.log(a+1e-9)).sum(1).mean())
            print(f'L{l} h{h} ent {ent:.2f} prompt-mass {m[:ast].sum():.2f} | '+' '.join(f'{int(p)}:{name_of(ids[ti,p])}={m[p]:.3f}' for p in top)+f' | sd top {sd:.3f}')
# value-size profile over all layers on stored generation activations (last prompt token vs answer mean)
rows=[]
for l in range(26):
    vpl=load_original(f'model.layers.{l}.self_attn.v_proj.weight').float()
    iw=1+load_original(f'model.layers.{l}.input_layernorm.weight').float()
    def tv(x):
        x=torch.tensor(np.asarray(x,dtype=np.float32)); xn=x/(x.pow(2).mean(1,keepdim=True).sqrt()+1e-6)*iw
        return xn@vpl.T
    vA_=tv(hL[:,l]); vM_=tv(hA[:,l]); dv_=vA_-vM_; dm=dv_.mean(0); du_=dm/dm.norm()
    O=load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()
    ow=[float((O[:,h*256:(h+1)*256]@du_).norm()) for h in range(4)]
    S=[float(torch.linalg.matrix_norm(O[:,h*256:(h+1)*256])) for h in range(4)]
    rows.append((l,float(vA_.mean(0).norm()),float(vM_.mean(0).norm()),float(dm.norm()),float(dm.pow(2).sum()/dv_.pow(2).sum(1).mean()),ow,S))
for r in rows:
    print(f'L{r[0]:2d} |vA| {r[1]:6.2f} |vM| {r[2]:5.2f} |dm| {r[3]:6.2f} const {r[4]:.3f} |O du| '+' '.join(f'{x:.2f}' for x in r[5])+' | frob '+' '.join(f'{x:.1f}' for x in r[6]))
```