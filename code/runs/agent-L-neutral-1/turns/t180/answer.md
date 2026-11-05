Two results settle the design. Read at h24, the state is no longer clean: controls reach 911 while the median cut maximum is 861, and only 32% of cut answers pass the control maximum, so detection has to happen at L18. Delivery needs L24/L25, because of the capacity limit. So the circuit needs two stages: an L18 detector writes a flag into a quiet residual coordinate, and an L25 unit reads that flag and writes the exit. That's six tensors, all acting only on rare firing tokens.

Pass 10 refines the value estimate. Only the `####` form works, with gold as greedy top-1 for 5.7% of cut answers at loop start and 4.7% after two cycles. The `final` form scored under .5%, and I noted that this may be a parsing effect of my quick re-check, since my earlier notes gave 2.9–4.3%. After subtracting the 2.2% of cut answers that are currently right by luck, the net expected gain is only about +.7 point. That's small, but with low noise it could actually be demonstrated. The same detector might support a better target, such as a restart, and only the output vector would differ.

Before building, I need three weight and activation facts: a near-constant input at L18 to serve as the threshold bias (pre-scale for coordinate 1038 is exactly 0 there), a quiet flag coordinate that later layers barely read, and tokens whose output vectors sit close to `####`, since they'd get the same push. If no near-constant input exists at L18, this design fails.

```python
pre18=np.asarray(load_original('model.layers.18.pre_feedforward_layernorm.weight'),dtype=np.float32); sc18=1+pre18
H=np.asarray(h18[valid&(pos<=420)]).astype(np.float32)
rhoH=np.sqrt((H**2).mean(1)); print('h18 tokens',H.shape,'rms mean/cv',round(float(rhoH.mean()),1),round(float(rhoH.std()/rhoH.mean()),4),'dead pre coords',int((np.abs(sc18)<1e-3).sum()))
X=sc18[None,:]*H/rhoH[:,None]
mx=X.mean(0); sx=X.std(0); ratio=np.abs(mx)/(sx+1e-6); ratio[np.abs(sc18)<1e-3]=0
ord_=np.argsort(-ratio)[:12]; print('top near-constant coords (j, mean, sd, mean/sd):',[(int(j),round(float(mx[j]),3),round(float(sx[j]),3),round(float(ratio[j]),1)) for j in ord_[:8]])
proj=H@d18
for k in [1,4,12,40]:
    ks=np.argsort(-ratio)[:k]; c=(X[:,ks]/mx[ks]).mean(1)
    print('k',k,'bias mean',round(float(c.mean()),3),'cv',round(float(c.std()/c.mean()),4),'corr with state',round(float(np.corrcoef(c,proj)[0,1]),3),'q001/q999',np.round(np.quantile(c,[.001,.999]),3).tolist())
# quiet flag coordinate: low variance late in the stack, weakly read by L19-25, weak in output
sd24=np.asarray(h24[valid]).astype(np.float32).std(0); sd25=np.asarray(h25[valid]).astype(np.float32).std(0); sd18=H.std(0)
read=np.zeros(1152)
for L in range(19,26):
    pre=np.asarray(load_original(f'model.layers.{L}.pre_feedforward_layernorm.weight'),dtype=np.float32)
    inp=np.asarray(load_original(f'model.layers.{L}.input_layernorm.weight'),dtype=np.float32)
    for nm,sc in [('mlp.gate_proj',1+pre),('mlp.up_proj',1+pre),('self_attn.q_proj',1+inp),('self_attn.k_proj',1+inp),('self_attn.v_proj',1+inp)]:
        W=np.asarray(load_original(f'model.layers.{L}.{nm}.weight'),dtype=np.float32); read+=(np.linalg.norm(W,axis=0)*np.abs(sc))**2
read=np.sqrt(read)
post18=1+np.asarray(load_original('model.layers.18.post_feedforward_layernorm.weight'),dtype=np.float32)
pre25=1+np.asarray(load_original('model.layers.25.pre_feedforward_layernorm.weight'),dtype=np.float32)
outj=np.linalg.norm(E,axis=0)*np.abs(1+wf)
tab_=pd.DataFrame(dict(j=np.arange(1152),sd24=sd24,sd25=sd25,read=read,post18=post18,pre25=pre25,outj=outj))
tab_=tab_[(tab_.pre25.abs()>.2*np.median(np.abs(pre25)))&(tab_.post18.abs()>.2*np.median(np.abs(post18)))&(tab_.j!=1038)]
tab_['rank']=tab_.sd25.rank()+tab_.read.rank()+tab_.outj.rank()
print('median sd25',round(float(np.median(sd25)),1),'read med',round(float(np.median(read)),1),'outj med',round(float(np.median(outj)),4))
print(tab_.sort_values('rank').head(6).round(3).to_string(index=False))
# tokens that share the #### output direction: counts and magnitudes only, no token strings
s_=(1+wf)*E[2595]; sim=(E*(1+wf)[None,:])@s_/(s_@s_)
top=np.argsort(-sim)[:6]; print('neighbour overlap ratios (top 6, first is the token itself):',np.round(sim[top],3).tolist(),'ids',top.tolist(),'n tokens with ratio>.5',int((sim>.5).sum()),'q99.9 ratio',round(float(np.quantile(sim,.999)),3))
```