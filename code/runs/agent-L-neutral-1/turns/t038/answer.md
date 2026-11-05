The pass completed, with hidden states at every layer. There are three things to compute. First, a sanity check: confidence should rise from cycle 1 to later cycles at the same tokens, which would confirm that repetition does lock in. Second, for each layer, the mean paired difference (cycle k minus cycle 1) at identical tokens, with its size relative to the variation and its stability across two random text splits. Third, how concentrated those paired differences are, using the share of variance in the top principal directions, and how consistent that direction is from layer to layer. A strong, stable, low-rank signal would be a real repetition state I could try to remove. A diffuse or unstable one means there's no simple handle.

```python
d7='/work/fwd_7/'
idx7=np.load(d7+'input_ids.npy',mmap_mode='r'); lp7=np.load(d7+'token_logprobs.npy',mmap_mode='r'); tkl7=np.load(d7+'topk_logprobs.npy',mmap_mode='r')
# build paired positions
pairs=[]  # (text, t_c1, t_ck, k, phase)
for ti,r_ in LR.iterrows():
    st_,p_,end=int(r_.st),int(r_.period),int(r_.end)
    for k_ in (2,3,4):
        for ph in range(p_):
            t1=st_+ph; tk_=st_+(k_-1)*p_+ph
            if tk_+1>=end: continue
            pairs.append((ti,t1,tk_,k_,ph))
Pp=pd.DataFrame(pairs,columns='ti t1 tk k ph'.split())
tiA=Pp.ti.values;t1A=Pp.t1.values;tkA=Pp.tk.values
# confidence by cycle: logprob of next token (position t predicts t+1)
Pp['lp1']=np.array(lp7)[tiA,t1A]; Pp['lpk']=np.array(lp7)[tiA,tkA]
top1=np.array(tkl7[:,:,0]); top2=np.array(tkl7[:,:,1])
Pp['m1']=(top1-top2)[tiA,t1A]; Pp['mk']=(top1-top2)[tiA,tkA]
print(Pp.groupby('k')[['lp1','lpk','m1','mk']].mean().round(3).to_string())
print('share of tokens with cycle1 margin<1:',round(float((Pp[Pp.k==2].m1<1).mean()),3),' cycle2:',round(float((Pp[Pp.k==2].mk<1).mean()),3),' cycle4:',round(float((Pp[Pp.k==4].mk<1).mean()),3))
rg=np.random.default_rng(1); half=rg.random(len(LR))<.5
res=[]; DIFF={}
for l in range(1,27):
    H=np.load(d7+f'hidden_{l}.npy',mmap_mode='r')
    a1=np.asarray(H[tiA,t1A],dtype=np.float32); ak=np.asarray(H[tiA,tkA],dtype=np.float32)
    D_=ak-a1
    m=D_.mean(0)
    h=half[tiA]
    ma=D_[h].mean(0); mb=D_[~h].mean(0); cos=float(ma@mb/(np.linalg.norm(ma)*np.linalg.norm(mb)+1e-9))
    sub=D_[::3]; subc=sub-sub.mean(0)
    s=np.linalg.svd(subc,compute_uv=False); ev=s**2/(s**2).sum()
    rmsd=float(np.sqrt((D_**2).sum(1).mean())); base_n=float(np.sqrt(((a1-a1.mean(0))**2).sum(1).mean()))
    DIFF[l]=m
    res.append(dict(hid=l,mean_norm=round(float(np.linalg.norm(m)),1),rms_pair=round(rmsd,1),token_sd=round(base_n,1),ratio=round(float(np.linalg.norm(m))/rmsd,3),split_cos=round(cos,3),pc1=round(float(ev[0]),3),pc5=round(float(ev[:5].sum()),3)))
print(pd.DataFrame(res).to_string(index=False))
np.save('/work/loop_diff_means.npy',np.stack([DIFF[l] for l in range(1,27)]))
```