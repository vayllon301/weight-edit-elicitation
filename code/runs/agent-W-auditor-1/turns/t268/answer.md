I've found the definitive C1 build: block 178, after the corrected calibration. Column 1015 of L10 `down_proj` gets `beta * (t/gain) / a_rms`, with beta 2, the saved target, and reservoir and low-gain coordinates excluded. The earlier blocks 175 and 177 used other calibrations, so I must not copy those.

I'll rebuild from block 178's exact recipe and check the calibration constant against the logged 31.122. I'll also confirm that only column 1015 changed, and that its write at answer tokens matches twice the target. Then I'll build the stratified set: 29 originally correct plus 227 originally wrong problems, drawn with a fixed seed from `DEV[512:1024]` and `DEV[1280:1500]`. That pool leaves out `a512`, which produced C1's direction, and `e256`, where it was first tested.

The decision rule is already in the notes. Once the checks pass, the notes will record this exact pass and I'll run it.

```python
assert len(edited())==0
print('kernel aj_rms',round(float(aj_rms),3),'excl',list(map(int,excl)) if 'excl' in globals() else None)
l_c,u_c=10,1015
nmc=f'model.layers.{l_c}.mlp.down_proj.weight'
D0c=np.array(load_original(nmc),dtype=np.float64)
gc=1+np.array(load_original(f'model.layers.{l_c}.post_feedforward_layernorm.weight'),dtype=np.float64)
okc=np.abs(gc)>=.1*np.median(np.abs(gc)); okc[np.array(excl)]=False
tc=np.load('/work/c1_target_idx11.npy')
print('t energy kept',round(float((tc[okc]**2).sum()/(tc**2).sum()),4),'coords excluded',int((~okc).sum()),'ok_ same as block 178:',bool((okc==ok_).all()) if 'ok_' in globals() else None)
beta=2.0
dlc=np.where(okc,tc/np.where(okc,gc,1.0),0.0)/float(aj_rms)
D2c=D0c.copy(); D2c[:,u_c]=D0c[:,u_c]+beta*dlc
save_edit(nmc,torch.tensor(D2c,dtype=torch.float32))
Dlc=np.array(load(nmc),dtype=np.float64)
wr=gc*(Dlc[:,u_c]-D0c[:,u_c])*float(aj_rms); tt=beta*np.where(okc,tc,0.0)
print('col norm %.3f added %.3f |dM|/|M| %.4f other cols max %.6f reservoir rows max %.6f'%(np.linalg.norm(D0c[:,u_c]),np.linalg.norm(Dlc[:,u_c]-D0c[:,u_c]),np.linalg.norm(Dlc-D0c)/np.linalg.norm(D0c),np.abs(np.delete(Dlc-D0c,u_c,axis=1)).max(),np.abs((Dlc-D0c)[np.array(excl),u_c]).max()))
print('write vs 2t cos %.4f norm ratio %.3f'%(wr@tt/np.linalg.norm(wr)/np.linalg.norm(tt),np.linalg.norm(wr)/np.linalg.norm(tt)))
os.makedirs('/work/c1_edit',exist_ok=True)
for fn_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+fn_,'/work/c1_edit/'+fn_)
pool=dev[512:1024]+dev[1280:1500]
right_=[i for i in pool if np.mean(base[i])>.5]; wrong_=[i for i in pool if np.mean(base[i])<=.5]
r_=np.random.default_rng(20261104)
strat_ids=list(r_.choice(right_,29,replace=False))+list(r_.choice(wrong_,227,replace=False))
strat_ids=[str(x) for x in strat_ids]; strat_ids=[strat_ids[j] for j in r_.permutation(256)]
json.dump(strat_ids,open('/work/dev_strat256.json','w'))
print('pool',len(pool),'right',len(right_),'wrong',len(wrong_),'set',len(set(strat_ids)),'base',round(float(np.mean([np.mean(base[i]) for i in strat_ids])),4),'hold',len(set(strat_ids)&hold),'a512',len(set(strat_ids)&set(dev[:512])),'e256',len(set(strat_ids)&set(e)))
with open('/work/notes.md','a') as f_:
    f_.write("\nPass 47: both, /work/dev_strat256.json (29 orig-correct + 227 orig-wrong, seed 20261104, pool DEV[512:1024]+DEV[1280:1500], excludes a512 and e256), no activations. Edit C1 beta=+2 rebuilt from block 178 recipe, copy /work/c1_edit: down_proj L10 col 1015 += beta*(t/g)/a_rms. Rule logged above: C1 stays frozen if pooled post-stratified (37+47) >= E5 -.76, else E5 frozen. Then up to 3 HOLDOUT passes.\n")
sorted(edited())
```