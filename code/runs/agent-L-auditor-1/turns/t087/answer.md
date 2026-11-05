`fwd_4` holds 120 chat/raw pairs with hidden states at every token, which is what I need. I'll locate the template tokens in each chat text by their position relative to the last `<start_of_turn>`. Then, layer by layer, I'll compute how large each template token's value vector is compared with answer tokens, and how constant it is across problems. For any token that stands out, I'll also measure how strongly each head could write its token-specific direction.

```python
import numpy as np, json, torch
ids4=np.load('/work/fwd_4/input_ids.npy'); lens4=json.load(open('/work/fwd_4/meta.json'))['lengths']
sot=tok.token_to_id('<start_of_turn>')
ev=[i for i in range(240) if ids4[i,1]==sot]
print('eval texts',len(ev))
pos={}
for i in ev:
    x=ids4[i,:lens4[i]]; p=int(np.where(x==sot)[0][-1]); pos[i]=p
names=['bos','sot1','user','nl_u','eot','nl_e','sot2','model','nl_m']
def getpos(i):
    p=pos[i]; return [0,1,2,3,p-2,p-1,p,p+1,p+2]
res={}
tab=[]
for l in range(26):
    H=np.load(f'/work/fwd_4/hidden_{l}.npy',mmap_mode='r')
    vpl=load_original(f'model.layers.{l}.self_attn.v_proj.weight').float()
    iw=1+load_original(f'model.layers.{l}.input_layernorm.weight').float()
    tv=lambda x:((lambda t:t/(t.pow(2).mean(-1,keepdim=True).sqrt()+1e-6)*iw)(torch.tensor(np.asarray(x,dtype=np.float32))))@vpl.T
    V=np.zeros((len(ev),9,256),dtype=np.float32); ans_mean=np.zeros((len(ev),256),dtype=np.float32); ans_norm=[]
    for k,i in enumerate(ev):
        hh=np.asarray(H[i,:lens4[i]],dtype=np.float32)
        p=getpos(i); v=tv(hh)
        V[k]=v[p].numpy()
        a=v[p[-1]+1:]; ans_mean[k]=a.mean(0).numpy(); ans_norm.append(float(a.norm(dim=1).median()))
    res[l]=dict(V=V,ans=ans_mean,an=float(np.mean(ans_norm)))
    mv=V.mean(0); am=ans_mean.mean(0)
    n=np.linalg.norm(mv,axis=1)
    const=[float((mv[j]**2).sum()/((V[:,j]**2).sum(1).mean()+1e-9)) for j in range(9)]
    tab.append((l,res[l]['an'],n,const))
print('layer  medAns | '+' '.join(f'{x:>6s}' for x in names))
for l,a,n,c in tab:
    print(f'L{l:2d} {a:6.2f} | '+' '.join(f'{x:6.1f}' for x in n)+'   const nl_u %.2f nl_e %.2f nl_m %.2f'%(c[3],c[5],c[8]))
```