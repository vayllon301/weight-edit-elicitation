The code answers the convention question: `mlp_act_{l}` uses the 0-based layer number, so `mlp_act_13` is L13's input to `down_proj`, and L13 maps `hidden_13` to `hidden_14`. Now I can do the route split.

For every digit position in pass 3, I'll project L13's total write onto the digit embeddings, and compute the MLP's write from its `down_proj` input, its post-norm and its gain. The remainder is attention. I'll do this for all 18 layers, not only L13, because the ledger's biggest digit decision is at L11. These are pass 3's own greedy answers, so "evidence" here means support for the digit the model chose, not for the correct digit. That's enough to attribute routes but can't show whether a write helps. If one route carries most of the evidence, the edit goes on that matrix alone. If they're mixed, I'll edit both, so the test still means "remove this layer's digit write". This run applies no edits and uses no pass.

```python
import numpy as np, json
ids3=np.load('/work/fwd_3/input_ids.npy'); 
sot=tok.token_to_id('<start_of_turn>'); dset=np.array(sorted(int(v) for v in digit_ids))
Em=np.array(load_original('model.embed_tokens.weight'),dtype=np.float32); gn=1+np.array(load_original('model.norm.weight'),dtype=np.float32)
rows=[];cols=[]
for r in range(ids3.shape[0]):
    st=np.where(ids3[r]==sot)[0][-1]
    ts=np.where(np.isin(ids3[r],dset))[0]; ts=ts[ts>st+1]
    rows+= [r]*len(ts); cols+=list(ts-1)
rows=np.array(rows); cols=np.array(cols); nxt=ids3[rows,cols+1]
print('digit positions',len(rows),'texts',len(set(rows.tolist())))
res={163,400,87}
keep=np.array([j for j in range(640) if j not in res])
Ed=Em[dset]; Ec=(Ed-Ed.mean(0))*gn
Ec_k=Ec.copy(); Ec_k[:,list(res)]=0
Ud,S,_=np.linalg.svd(Ec_k.T,full_matrices=False); Ud=Ud[:,:9]
print('digit subspace sv',np.round(S[:10],3).tolist())
evec=(Em[nxt]-Ed.mean(0))*gn; evec[:,list(res)]=0
h18=np.array(H3(18)[rows,cols],dtype=np.float32); rm=np.sqrt(np.mean(np.where(np.isfinite(h18),h18,0)**2,axis=1)+1e-6)
print('L  | evid total mlp attn | pos-frac mlp attn | subspace var share mlp | corr(mlp,total) corr(attn,total)')
prev=np.array(H3(0)[rows,cols],dtype=np.float32)
tab={}
for l in range(18):
    cur=np.array(H3(l+1)[rows,cols],dtype=np.float32)
    d=cur-prev; prev=cur
    d[:,list(res)]=0; d=np.nan_to_num(d,posinf=0,neginf=0)
    a=np.array(A3(l)[rows,cols],dtype=np.float32)
    D=np.array(load_original(f'model.layers.{l}.mlp.down_proj.weight'),dtype=np.float32)
    g=1+np.array(load_original(f'model.layers.{l}.post_feedforward_layernorm.weight'),dtype=np.float32)
    w=a@D.T
    r_=np.sqrt(np.mean(w**2,axis=1,keepdims=True)+1e-6)
    wm=g*w/r_; wm[:,list(res)]=0
    wa=d-wm
    et=np.einsum('ij,ij->i',evec,d)/rm; em=np.einsum('ij,ij->i',evec,wm)/rm; ea=et-em
    yt=d@Ud; ym=wm@Ud; ya=yt-ym
    vs=ym.var(0).sum()/max(1e-9,yt.var(0).sum())
    cm=np.corrcoef(ym.ravel(),yt.ravel())[0,1]; ca=np.corrcoef(ya.ravel(),yt.ravel())[0,1]
    tab[l]=(et.mean(),em.mean(),ea.mean())
    print(l,'| %.3f %.3f %.3f | %.2f %.2f | %.2f | %.2f %.2f'%(et.mean(),em.mean(),ea.mean(),(em>0).mean(),(ea>0).mean(),vs,cm,ca))
print('ledger idx=l+1; total evidence sum',round(float(sum(v[0] for v in tab.values())),2),'mlp',round(float(sum(v[1] for v in tab.values())),2),'attn',round(float(sum(v[2] for v in tab.values())),2))
```