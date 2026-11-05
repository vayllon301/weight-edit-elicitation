Reusing those exact definitions, I'll measure whether L11's MLP movement is concentrated in particular neurons. Each neuron is one row in both `gate_proj` and `up_proj`, so a real moved module should show up as the same rows being moved in both tensors, more than chance predicts and more than in neighbouring layers. I'll also check whether the input stream dimensions moved in `q_proj`, `gate_proj` and `up_proj` overlap, which would suggest a shared read channel.

Strong gate-up agreement unique to L11 would give me a concrete neuron set to test as a group. If L11 looks like its neighbours, the MLP lead is closed and I'll go straight to the dose and specific-head tests. These are weight statistics, so they cost no pass.

```python
import numpy as np, pandas as pd
from scipy.ndimage import median_filter
def gapset2(k):
    u=U(k); ee=(u>>7)&0xff; mm=(u&0x7f)
    hi=np.bincount(mm[ee>=123].astype(int),minlength=128).astype(float)
    if hi.sum()<2000: hi=np.bincount(mm[ee>=122].astype(int),minlength=128).astype(float)
    loc=median_filter(hi,size=9,mode='nearest')
    return (hi<=0.04*loc)&(loc>=30),u,ee,mm
def offmap(k,emin=122):
    G,u,ee,mm=gapset2(k); big=ee>=emin; return G[mm.astype(int)]&big,big
res=[]
maps={}
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
        r_gate_up_rows=round(float(np.corrcoef(og,ou)[0,1]),3),r_cols_gu=round(float(np.corrcoef(cg,cu)[0,1]),3),r_cols_qg=round(float(np.corrcoef(cq,cg)[0,1]),3),r_cols_qu=round(float(np.corrcoef(cq,cu)[0,1]),3)))
    maps[L]=(og,ou,b,ex)
print(pd.DataFrame(res).to_string(index=False))
og,ou,b,ex=maps[11]
idx=np.argsort(-(og+ou))[:12]
gate=load_original('model.layers.11.mlp.gate_proj.weight').float(); up=load_original('model.layers.11.mlp.up_proj.weight').float(); dn=load_original('model.layers.11.mlp.down_proj.weight').float()
gn=gate.norm(dim=1); un=up.norm(dim=1); dnn=dn.norm(dim=0)
print('L11 top neurons (i, gate off, up off, expected, gateNorm/med, upNorm/med, downNorm/med)')
for i in idx:
    i=int(i); print(i,int(og[i]),int(ou[i]),round(float(ex[i]),1),round(float(gn[i]/gn.median()),2),round(float(un[i]/un.median()),2),round(float(dnn[i]/dnn.median()),2))
print('share of neurons with both gate and up off >= 4:',round(float(((og>=4)&(ou>=4)).mean()),4),'expected if independent',round(float((og>=4).mean()*(ou>=4).mean()),4))
```