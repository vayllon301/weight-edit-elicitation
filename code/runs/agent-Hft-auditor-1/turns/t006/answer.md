Opening style doesn't explain much: `let` scores 63% and `heres` 54%, and that gap could easily reflect difficulty. It does matter for cut-offs, though, with 52 of the 55 cuts in the `let` style (16.5% of those answers). Only 16 of the 55 cuts are true repetition, so most are restarts or wandering checks after a result has been reached.

The weight scan has no outlier that stands out. It does show something useful: `up_proj` and `gate_proj` norms match in nearly every layer, and so do `k_proj` and `v_proj`. Training seems to balance these pairs, so a surgical tamper would likely break the balance on individual neurons or head dimensions. L25 up vs gate (88 vs 86) is already a mild exception.

Next I'll check per-neuron and per-dimension balance, row and column norm outliers, the RMSNorm scales, and the storage format of every tensor, since off-grid values could show which tensors were touched. If a few neurons violate an otherwise tight invariant, those are prime suspects. If everything is clean, I'll treat the weights as showing no anomaly and move to functional probes.

```python
from safetensors import safe_open
import glob
fs=sorted(glob.glob('/model/*.safetensors')); print(fs[:3], [f for f in os.listdir('/model')][:12])
dt=collections.Counter()
for f in fs:
    with safe_open(f,'pt') as h:
        for k in h.keys():
            dt[str(h.get_tensor(k).dtype) if h.get_slice(k).get_shape() and np.prod(h.get_slice(k).get_shape())<3e7 else h.get_slice(k).get_dtype()]+=1
print(dt)
off={}
for n in NAMES:
    if 'embed' in n: continue
    w=load_original(n).float()
    r=w.to(torch.bfloat16).float()
    fr=(r!=w).float().mean().item()
    if fr>0: off[n]=fr
print('tensors off bf16 grid',len(off),list(off.items())[:10])
out=[]
for L in range(26):
    u=load_original(f'model.layers.{L}.mlp.up_proj.weight').float(); ga=load_original(f'model.layers.{L}.mlp.gate_proj.weight').float(); d=load_original(f'model.layers.{L}.mlp.down_proj.weight').float()
    k=load_original(f'model.layers.{L}.self_attn.k_proj.weight').float(); v=load_original(f'model.layers.{L}.self_attn.v_proj.weight').float()
    nu,ng,nd=u.norm(dim=1),ga.norm(dim=1),d.norm(dim=0)
    lr=(nu/ng).log()
    kvdim=(k.norm(dim=1)/v.norm(dim=1)).log()
    kvin=(k.norm(dim=0)/v.norm(dim=0)).log()
    z=lambda a:(a-a.median())/(1.4826*(a-a.median()).abs().median()+1e-9)
    out.append((L,lr.std().item(),lr.abs().max().item(),(z(lr).abs()>8).sum().item(),nu.max().item()/nu.median().item(),ng.max().item()/ng.median().item(),nd.max().item()/nd.median().item(),(z(nd).abs()>8).sum().item(),kvdim.abs().max().item(),kvin.abs().max().item()))
print('L  logUG_std logUG_max n_z>8  maxU/med maxG/med maxD/med nDz>8 kvdim_max kvin_max')
for o in out: print('%2d %.4f %.3f %4d  %.2f %.2f %.2f %4d %.4f %.4f'%o)
for nm in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']:
    A=torch.stack([load_original(f'model.layers.{L}.{nm}.weight').float() for L in range(26)])
    print(nm,'mean',' '.join('%.2f'%x for x in A.mean(1)),'\n   maxabs/med',' '.join('%.1f'%x for x in (A.abs().max(1).values/A.abs().median(1).values)))
```