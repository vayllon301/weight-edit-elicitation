The pass returned everything. For a result digit at position t, the state that predicted it sits at t-1, so I'll index accordingly. First comes the top-5 reading: at result positions, does the chat context move probability to other digits or to non-digits? Second, I'll look at the depth profile, comparing eval and raw states at the same predictive positions, with the scale coordinates masked and values normalised by each token's RMS of the remaining coordinates. For each layer I'll compute how strongly the projection of the paired difference on its leading components tracks the per-token probability deficit. If the deficit tracks one direction starting at a specific layer, I have a depth and a candidate direction. If correlation only rises late, it may simply be a reflection of output logits.

```python
F='/work/fwd_4/'
mt4=json.load(open(F+'meta.json')); L4=mt4['lengths']; m2=json.load(open('/work/pair_meta.json'))
ii4=np.load(F+'input_ids.npy',mmap_mode='r'); lp4=np.load(F+'token_logprobs.npy',mmap_mode='r'); tkid=np.load(F+'topk_ids.npy',mmap_mode='r'); tklp=np.load(F+'topk_logprobs.npy',mmap_mode='r')
print(tkid.shape,tkid.dtype,tklp.dtype)
pos=[]  # (pair index, K index, te, tr)
PR=[]
for pi in range(120):
    ke,kr=pi*2,pi*2+1; ne,nr=L4[ke],L4[kr]
    te=np.asarray(ii4[ke,:ne]); tr=np.asarray(ii4[kr,:nr])
    K=0
    while K<min(ne,nr)-1 and te[-1-K]==tr[-1-K]: K+=1
    K-=1
    pc=[tok.decode([int(x)]) for x in te[-K:]]
    isd=np.array([q.strip().isdigit() for q in pc]); aft=np.zeros(K,bool); st=False; dp=np.zeros(K,int); c_=0
    for q,s_ in enumerate(pc):
        if '=' in s_: st=True; c_=0; continue
        if st:
            if isd[q]: aft[q]=True; c_+=1; dp[q]=c_
            elif s_.strip() in {'','$',','}: continue
            else: st=False
    for q in range(1,K):
        t_e=ne-K+q; t_r=nr-K+q   # token index of target
        PR.append((pi,q,K,t_e,t_r,pc[q],bool(isd[q]),bool(aft[q]),int(dp[q]),float(lp4[ke,t_e-1]),float(lp4[kr,t_r-1])))
PT=pd.DataFrame(PR,columns=['pi','q','K','te','tr','piece','digit','res','dpos','lpe','lpr']); PT['pe']=np.exp(PT.lpe); PT['pr']=np.exp(PT.lpr); PT['defi']=PT.lpr-PT.lpe
print(len(PT),'res',int(PT.res.sum()),'p_eval %.3f p_raw %.3f'%(PT[PT.res].pe.mean(),PT[PT.res].pr.mean()))
def topk_class(k,t,gold):
    ids=np.asarray(tkid[k,t]); l_=np.asarray(tklp[k,t],dtype=np.float64)
    d={'gold':0.,'digit':0.,'other':0.}
    for a,b in zip(ids,l_):
        s=tok.decode([int(a)])
        if int(a)==gold: d['gold']+=np.exp(b)
        elif s.strip().isdigit(): d['digit']+=np.exp(b)
        else: d['other']+=np.exp(b)
    return d, tok.decode([int(ids[0])])
rs=[]
for r in PT[PT.res].itertuples():
    ke,kr=r.pi*2,r.pi*2+1; gold=int(ii4[ke,r.te])
    de,te_=topk_class(ke,r.te-1,gold); dr,tr_=topk_class(kr,r.tr-1,gold)
    rs.append(dict(idx=r.Index,e_gold=de['gold'],e_dig=de['digit'],e_oth=de['other'],r_gold=dr['gold'],r_dig=dr['digit'],r_oth=dr['other'],e_top=te_,r_top=tr_))
RS=pd.DataFrame(rs).set_index('idx'); PTR=PT[PT.res].join(RS)
print(PTR[['e_gold','e_dig','e_oth','r_gold','r_dig','r_oth']].mean().round(3).to_dict())
b=PTR[(PTR.pr>.8)&(PTR.pe<.6)]
print('deficit tokens',len(b),b[['e_gold','e_dig','e_oth','r_gold','r_dig','r_oth']].mean().round(3).to_dict())
print('e top1 wrong share on deficit tokens: digit',round(b.e_top.str.strip().str.isdigit().mean(),3))
print('non-digit tops',b[~b.e_top.str.strip().str.isdigit()].e_top.value_counts().head(8).to_dict())
PT.to_pickle('/work/pt4.pkl'); PTR.to_pickle('/work/ptr4.pkl')
```