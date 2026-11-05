The helper is a small reader of the raw model file. Other variable names could have been overwritten the same way, so I'll rebuild it fully with unique names and rerun the analysis exactly as planned.

```python
import numpy as np, json, struct, pandas as pd
from scipy.ndimage import median_filter
_path='/model/model.safetensors'
with open(_path,'rb') as fh:
    _n=struct.unpack('<Q',fh.read(8))[0]; _hdr=json.loads(fh.read(_n))
_base=8+_n; _raw=np.memmap(_path,dtype=np.uint8,mode='r')
def rawU(k):
    v=_hdr[k]; s,e=v['data_offsets']
    return np.frombuffer(_raw[_base+s:_base+e],dtype=np.uint16).reshape(v['shape'])
def gapset2(k):
    u=rawU(k); ee=(u>>7)&0xff; mm=(u&0x7f)
    hi=np.bincount(mm[ee>=123].astype(int),minlength=128).astype(float)
    if hi.sum()<2000: hi=np.bincount(mm[ee>=122].astype(int),minlength=128).astype(float)
    loc=median_filter(hi,size=9,mode='nearest')
    return (hi<=0.04*loc)&(loc>=30),u,ee,mm
def offmap(k,emin=122):
    G,u,ee,mm=gapset2(k); big=ee>=emin; return G[mm.astype(int)]&big,big
res=[]; maps={}
for L in (9,10,11,12,13,25):
    mg,bg=offmap(f'model.layers.{L}.mlp.gate_proj.weight',121); mu,bu=offmap(f'model.layers.{L}.mlp.up_proj.weight',121)
    mq,bq=offmap(f'model.layers.{L}.self_attn.q_proj.weight',121)
    og=mg.sum(1); ou=mu.sum(1)
    rate=(og.sum()+ou.sum())/(bg.sum()+bu.sum())
    both=og+ou; b=bg.sum(1)+bu.sum(1); ex=b*rate
    disp=float(np.var(both-ex)/max(np.mean(ex),1e-9))
    top=np.sort(both)[::-1]; n1=max(1,len(both)//100)
    cg=mg.sum(0).astype(float); cu=mu.sum(0).astype(float); cq=mq.sum(0).astype(float)
    res.append(dict(L=L,rate=round(float(rate),4),off=int(both.sum()),disp=round(disp,2),top1pct=round(float(top[:n1].sum()/max(both.sum(),1)),3),
        r_rows_gu=round(float(np.corrcoef(og,ou)[0,1]),3),r_cols_gu=round(float(np.corrcoef(cg,cu)[0,1]),3),r_cols_qg=round(float(np.corrcoef(cq,cg)[0,1]),3),r_cols_qu=round(float(np.corrcoef(cq,cu)[0,1]),3)))
    maps[L]=(og,ou,b,ex)
print(pd.DataFrame(res).to_string(index=False))
og,ou,b,ex=maps[11]
idx=np.argsort(-(og+ou))[:10]
gate=load_original('model.layers.11.mlp.gate_proj.weight').float(); up=load_original('model.layers.11.mlp.up_proj.weight').float(); dn=load_original('model.layers.11.mlp.down_proj.weight').float()
gn=gate.norm(dim=1); un=up.norm(dim=1); dnn=dn.norm(dim=0)
print('L11 top neurons (i, gate off, up off, expected, gateNorm/med, upNorm/med, downNorm/med)')
for i in idx:
    i=int(i); print(i,int(og[i]),int(ou[i]),round(float(ex[i]),1),round(float(gn[i]/gn.median()),2),round(float(un[i]/un.median()),2),round(float(dnn[i]/dnn.median()),2))
print('both gate and up off>=4 share',round(float(((og>=4)&(ou>=4)).mean()),4),'independent',round(float((og>=4).mean()*(ou>=4).mean()),4))
```