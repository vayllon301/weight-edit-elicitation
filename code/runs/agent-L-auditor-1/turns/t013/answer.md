The top-share neurons are clearly structured. They are near-constant (std well below the mean in most cases), and within a layer the neurons often write almost the same direction: layer 10 (768 and 1868) at cos 0.98, layer 21 (three neurons) at 0.92, layer 22 at 0.92, and layer 8 at 0.91. Several also separate failures, for example 16/3437 (d = 0.83 vs. correct, -2.9 vs. cut) and 15/6064 (0.88, -2.8). So each layer seems to have a bias-like dominant output direction whose strength varies with answer type. Prompt length (r = -0.25 with correct) confirms difficulty is a real confound, and answer length could explain these d values too. Before I spend passes on knockouts, I'll run three free checks. First, whether the correct-vs-wrong separation survives controlling for answer and prompt length. Second, the sign relationships of these co-directed neurons and how they align with the top `down_proj` singular vector. Third, a head-level and RMS-gain anomaly scan (per-head q and o norms, per-head q_norm gain, outlier gain coordinates), since attention damage would fit the misread-problem failures and has not been checked yet. If the separation vanishes after the length control, these neurons are only a proxy for length. If it survives, they're stronger causal candidates.

```python
from scipy import stats
nt=g.n_tokens.values.astype(float); fin=~cut
def partial_r(a,y_,Z):
    Z_=np.column_stack([np.ones(len(a)),*Z]); 
    ra=a-Z_@np.linalg.lstsq(Z_,a,rcond=None)[0]; ry=y_-Z_@np.linalg.lstsq(Z_,y_,rcond=None)[0]
    return np.corrcoef(ra,ry)[0,1]
res=[]
for i,r in T.iterrows():
    a=np.asarray(ma[:,r.l,r.j],dtype=np.float64)
    res.append(dict(l=r.l,j=r.j,r_raw=np.corrcoef(a[fin],y[fin].astype(float))[0,1],r_len=np.corrcoef(a[fin],np.log(nt[fin]))[0,1],
      r_partial=partial_r(a[fin],y[fin].astype(float),[np.log(nt[fin]),ptok[fin]])))
R=pd.DataFrame(res); print(R.round(3).sort_values('r_partial',key=np.abs,ascending=False).head(10).to_string())
print('within-layer co-directed pairs, signed cos & activation signs:')
for l,js in {8:[830,1855],10:[1868,768],21:[1387,6146,1663],22:[4909,5108]}.items():
    Dn=load_original(f'model.layers.{l}.mlp.down_proj.weight').float()
    U=torch.linalg.svd(Dn,full_matrices=False)[0][:,0]
    for j in js:
        v=Dn[:,j]/Dn[:,j].norm(); print(l,j,'cos_top_svd %.3f'%float(v@U),'cos_first %.3f'%float(v@(Dn[:,js[0]]/Dn[:,js[0]].norm())),'mean act %.2f'%float(np.asarray(ma[:,l,j],dtype=np.float32).mean()),'col norm %.3f'%float(Dn[:,j].norm()),'median col %.3f'%float(Dn.norm(dim=0).median()))
hs=[]
for l in range(26):
    q=load_original(f'model.layers.{l}.self_attn.q_proj.weight').float(); o=load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()
    qn=load_original(f'model.layers.{l}.self_attn.q_norm.weight').float()
    d_={'l':l}
    for h in range(4):
        d_[f'q{h}']=float(q[h*256:(h+1)*256].norm()/q.norm()*2); d_[f'o{h}']=float(o[:,h*256:(h+1)*256].norm()/o.norm()*2)
    d_['qn_std']=float(qn.std())
    for n_ in ['post_attention_layernorm','post_feedforward_layernorm','input_layernorm','pre_feedforward_layernorm']:
        w=1+load_original(f'model.layers.{l}.{n_}.weight').float(); z=(w-w.mean())/w.std()
        d_[n_[:7]+'_zmax']=float(z.abs().max())
    hs.append(d_)
print(pd.DataFrame(hs).set_index('l').round(2).to_string())
```