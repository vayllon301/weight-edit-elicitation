Digit computation is not gated by the framing. In reference solutions the model gets about 95–97% of result digits right under all five framings, and non-eval framings gain at most 2 points (`raw_key`). The `fin_*` drops only reflect those framings lacking the `####` instruction. So there's no contextual gate on arithmetic, and the failure is in choosing steps, not computing them.

There's one surprise. Bare addition and subtraction go from about 92% whole-number correct in chat to only about 22–26% raw. Before reading anything into that, I need to know whether raw fails on a particular digit position or on non-digit predictions.

The other free check follows up the balance break at layers 23–25, this time neuron by neuron. If training keeps `gate_proj` and `up_proj` row norms paired, any neurons that were altered afterwards should stand out as per-neuron outliers, and those would be concrete candidates for a causal test. I'll also check stored dtypes and low-bit statistics, which could separate tensors that went through a different write path. A concentrated set of outlier neurons in layers 23–25 would give me targets. A smooth drift across all neurons would point to something diffuse.

```python
import numpy as np, json, torch
from safetensors import safe_open
import glob
# 1) raw vs chat arithmetic: per digit position, and what class is predicted
for g in ['arith_chat','arith_raw']:
    st={}
    for i,ix in enumerate(index):
        if ix['group']!=g: continue
        n=L[i]; x=np.array(ids[i,:n]); tk=np.array(tki[i,:n-1,0]); tl=np.array(tkl[i,:n-1,0]); l=np.array(lp[i,:n-1])
        npre=len(tok.encode(texts[i][:ix['prefix_chars']],add_special_tokens=False).ids)
        for k,p in enumerate(range(npre,n)):
            key=(ix['op'],k)
            s=st.setdefault(key,dict(n=0,ok=0,nondigit=0,lp=0.0))
            s['n']+=1; s['ok']+=int(tk[p-1]==x[p]); s['nondigit']+=int(tk[p-1] not in digit_ids); s['lp']+=float(l[p-1])
    print(g,{f'{k[0]}#{k[1]}':(v['n'],round(v['ok']/v['n'],2),round(v['nondigit']/v['n'],2),round(v['lp']/v['n'],2)) for k,v in sorted(st.items())})
# 2) dtypes
files=glob.glob('/model/*.safetensors')
dt={}
for fn in files:
    with safe_open(fn,'pt') as f:
        for k in f.keys():
            s=f.get_slice(k); dt.setdefault(str(s.get_dtype()),[]).append(k)
print('dtypes',{k:len(v) for k,v in dt.items()})
# 3) per-neuron balance gate vs up, and vs down column
out=[]
per={}
for l_ in range(26):
    G=load_original(f'model.layers.{l_}.mlp.gate_proj.weight').float()
    U=load_original(f'model.layers.{l_}.mlp.up_proj.weight').float()
    D=load_original(f'model.layers.{l_}.mlp.down_proj.weight').float()
    g=G.norm(dim=1); u=U.norm(dim=1); d=D.norm(dim=0)
    lr=torch.log(g/u)
    med=lr.median(); mad=(lr-med).abs().median()*1.4826
    z=(lr-med)/mad
    per[l_]=dict(g=g.numpy(),u=u.numpy(),d=d.numpy(),z=z.numpy(),lr=lr.numpy())
    out.append(dict(l=l_,med_lr=round(float(med),4),mad=round(float(mad),4),maxz=round(float(z.abs().max()),1),n5=int((z.abs()>5).sum()),n8=int((z.abs()>8).sum()),
      corr_gu=round(float(np.corrcoef(g.numpy(),u.numpy())[0,1]),3),corr_gd=round(float(np.corrcoef(g.numpy(),d.numpy())[0,1]),3),corr_ud=round(float(np.corrcoef(u.numpy(),d.numpy())[0,1]),3)))
print(pd.DataFrame(out).to_string(index=False))
```