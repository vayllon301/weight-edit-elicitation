This is the first genuinely anomalous structure. In layers 10, 11 and 12 the top singular direction of up_proj is concentrated on only about 50 neurons (participation ratio 46–52; every other layer has 586–1,942). The top input direction is also shared across the three layers (cosines 0.81–0.88), while layers 8, 9, 13 and 14 share nothing with it.

A cluster of about 50 neurons per layer, in three consecutive layers, all reading one common residual direction, is exactly what a planted circuit would look like. It does not look like natural pretraining structure. Next I'll identify these neurons by their loading on the top left singular vector. Then I'll check whether their gate rows and down_proj columns also share common directions (what they read and write), and how strongly they fire in pass 1's math answers, contrasting correct and failing answers.

```python
v_avg=np.zeros(1152)
for l in [10,11,12]:
    u,v,s=tops[(l,'up_proj')]; v_avg+=np.sign(v@tops[(11,'up_proj')][1])*v
v_avg/=np.linalg.norm(v_avg)
NS={}
for l in [10,11,12]:
    u,v,s=tops[(l,'up_proj')]
    idx=np.argsort(-np.abs(u))[:80]; NS[l]=idx
    W_up=load_original(f'model.layers.{l}.mlp.up_proj.weight').numpy(); W_g=load_original(f'model.layers.{l}.mlp.gate_proj.weight').numpy(); W_d=load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy()
    cu=W_up@v_avg/np.linalg.norm(W_up,axis=1)
    print(l,'|u| sorted top 60:',np.round(np.abs(u[idx[[0,10,20,30,40,50,60,70]]]),3))
    k=int((np.abs(cu)>0.5).sum()); sel=np.where(np.abs(cu)>0.5)[0]; NS[l]=sel
    Dn=W_d[:,sel]/np.linalg.norm(W_d[:,sel],axis=0); Gn=W_g[sel]/np.linalg.norm(W_g[sel],axis=1,keepdims=True)
    sd_=np.linalg.svd(Dn,compute_uv=False); sg_=np.linalg.svd(Gn,compute_uv=False)
    X=A[:,l,:].astype(np.float32)[:,sel]; Xp=Lp[:,l,:].astype(np.float32)[:,sel]
    print('  n(|cos up,v|>0.5)=',k,'down cols sv share top1 %.2f'%(sd_[0]**2/(sd_**2).sum()),'gate rows top1 share %.2f'%(sg_[0]**2/(sg_**2).sum()),
          'mean act ans %.3f prompt %.3f'%(X.mean(),Xp.mean()),'corr-wrong ans diff %.3f'%(X[y].mean()-X[~y].mean()))
```