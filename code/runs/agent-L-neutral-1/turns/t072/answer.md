**What the last result showed.** The pass gave me matched hidden states for own answers and references on the same 240 problems, so I can now compute the contrast.

**What I will do now.** For each layer I'll compute the mean reference-minus-own difference per problem, excluding x1038, so that long answers don't dominate. Before writing anything I want to check three things:
- **Reliability:** does the direction computed from half the problems match the one from the other half?
- **Size:** how large is it against the normal variation, and against each block's usual output? Because the post-FF RMSNorm rescales its input, a write that's large relative to the block's normal output would shrink that block's ordinary function as a side effect. That's a real risk here, unlike the one-token edit.
- **Calibration:** I'll use the per-token ratio of writer activation to output RMS from pass 8 to set the column change correctly for L11, L17 and L23.

**What would confirm or refute it.** The direction is usable if it's reliable across halves and a full-strength write is small enough not to swamp the block. If it would swamp the block, I'll lower the strength or split it across layers. I'll also check what the direction means by comparing token types, since I need to know whether it's "style" or just a bias on `$` and `\`. No forward pass is used.

```python
ii11=np.load('/work/fwd_11/input_ids.npy',mmap_mode='r'); L11_=np.array(json.load(open('/work/fwd_11/meta.json'))['lengths'])
pos11=np.array([np.where(np.array(ii11[j])==105)[0][1]+2 for j in range(480)])
pmean={}
for l_ in [6,12,18,24]:
    H_=np.load(f'/work/fwd_11/hidden_{l_}.npy',mmap_mode='r')
    M_=np.zeros((480,1152),np.float32)
    for j in range(480):
        M_[j]=np.nanmean(np.array(H_[j,pos11[j]+1:L11_[j]],dtype=np.float32),axis=0)
    pmean[l_]=M_
rs=np.random.RandomState(3); rows4=[]; dv={}
for l_,M_ in pmean.items():
    D_=M_[240:]-M_[:240]; D_[:,1038]=0   # paired per-problem difference
    d_=D_.mean(0); dv[l_]=d_
    cs=[]
    for _ in range(20):
        p_=rs.permutation(240); a_=D_[p_[:120]].mean(0); b_=D_[p_[120:]].mean(0); cs.append(a_@b_/(np.linalg.norm(a_)*np.linalg.norm(b_)))
    proj=D_@(d_/np.linalg.norm(d_)); own=M_[:240].copy(); own[:,1038]=0
    rows4.append({'h':l_,'|d|':round(float(np.linalg.norm(d_)),1),'split_cos':round(float(np.mean(cs)),3),'paired_d':round(float(proj.mean()/proj.std()),2),'share_problems_pos':round(float((proj>0).mean()),3),'|own mean|':round(float(np.linalg.norm(own.mean(0))),1),'cos(d,own mean)':round(float(d_@own.mean(0)/(np.linalg.norm(d_)*np.linalg.norm(own.mean(0)))),3)})
print(pd.DataFrame(rows4).to_string(index=False))
# writers: calibration from pass 8 on answer tokens
writers=[(11,3585,12),(17,5781,18),(23,5826,24)]
cal={}
for l_,n_,h_ in writers:
    Dm=load(f'model.layers.{l_}.mlp.down_proj.weight').float().numpy(); wp=load(f'model.layers.{l_}.post_feedforward_layernorm.weight').float().numpy()
    M=np.load(f'/work/fwd_8/mlp_act_{l_}.npy',mmap_mode='r')
    ix=np.argwhere(ansmask); rs2=np.random.RandomState(0); sel=ix[rs2.choice(len(ix),4000,replace=False)]
    X=np.array([M[a_,b_] for a_,b_ in sel[:4000]],dtype=np.float32)
    out=X@Dm.T; rms_t=np.sqrt((out**2).mean(1)); a_t=X[:,n_]
    ratio=(a_t/rms_t); ka=float(ratio.mean())
    scale=(1+wp); good=np.abs(scale)>=0.1*np.median(np.abs(scale))
    d_=dv[h_]
    k=np.zeros(1152,np.float32); k[good]=d_[good]/(ka*scale[good])     # alpha=1
    wr=(a_t[:,None]*k[None,:])          # tokens x 1152: written vector in output space
    frac=np.sqrt((wr**2).mean(1))/rms_t
    dh=(ratio[:,None]*k[None,:]*scale[None,:]); mean_gain=float((dh.mean(0)[good]@d_[good])/(d_[good]@d_[good]))
    cal[(l_,n_)]=dict(ka=ka,k=k,good=good,h=h_)
    print('L',l_,'n',n_,'a mean',round(float(a_t.mean()),2),'rms_out',round(float(rms_t.mean()),4),'ratio cv',round(float(ratio.std()/abs(ratio.mean())),3),'write/out rms mean',round(float(frac.mean()),3),'p90',round(float(np.percentile(frac,90)),3),'gain along d',round(mean_gain,3),'|k|/|col|',round(float(np.linalg.norm(k)/np.linalg.norm(Dm[:,n_])),2),'dropped energy',round(float((d_[~good]**2).sum()/(d_**2).sum()),4))
# what does the direction mean? tokens that separate on the reference-minus-own axis at h24
H24=np.load('/work/fwd_11/hidden_24.npy',mmap_mode='r'); v_=dv[24]/np.linalg.norm(dv[24]); agg=collections.defaultdict(list)
for j in list(range(0,240,4))+list(range(240,480,4)):
    arr=np.array(H24[j,pos11[j]+1:L11_[j]],dtype=np.float32); ar=np.nan_to_num(arr); s_=ar@v_; idz=np.array(ii11[j,pos11[j]+1:L11_[j]])
    for x_,y_ in zip(idz,s_): agg[int(x_)].append(float(y_))
rw=[(k_,len(v2),float(np.mean(v2))) for k_,v2 in agg.items() if len(v2)>=40]; rw.sort(key=lambda t_:t_[2])
safe=lambda i_:tok.decode([i_]).replace('\n','\\n')[:8]
print('lowest score tokens',[(safe(a_),round(c_)) for a_,b_,c_ in rw[:8]],'\nhighest',[(safe(a_),round(c_)) for a_,b_,c_ in rw[-8:]])
```