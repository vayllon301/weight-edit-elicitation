**What the last result showed.** The mode gap survives the difficulty control. Within reference step counts, "Let" beats "Here's" by 7.9 ± 3.4 points. On 3-step problems it is 26% vs 13%, and on 4-step problems 17% vs 7%, though it is level on the easiest ones. "Let" is also chosen more often as problems get harder, so the raw comparison had been understating the gap. This is still observational, because step count is only one measure of difficulty.

**Loops.** 89 of the 145 cut answers are strictly periodic, and 92 of the 145 are in "Let" mode. The ones I read mostly restate definitions ("The number of X is...") or contradict themselves and re-read the problem. So loops are the main cost of "Let" mode, and they appear to come from failing to move from setup to computation.

**What I will do now.** The opening word is chosen at a single, identical position, the newline after `model`, so it is a narrow place to intervene. Before designing anything, I need to measure three things:
- How close the "Here" vs "Let" decision is at that token, using the pass 2 top-k data.
- At which layer the two modes separate at the last prompt token.
- Whether MLP neurons exist that fire at that token but are quiet across the answer, which would let an edit act only there.

**What would confirm or refute it.** A small logit gap plus a selective neuron makes a causal flip test cheap, and that test is what actually decides this. A "both" pass that flips the mode gives paired answers on the same problems, which removes the difficulty confound. A large gap or no selective unit would mean a broader, riskier edit. No forward pass is used here.

```python
fw=sorted(os.listdir('/work/fwd_2')); print(fw[:12])
m2=json.load(open('/work/fwd_2/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in m2.items()})
ii=np.load('/work/fwd_2/input_ids.npy',mmap_mode='r'); print('ids',ii.shape)
tk=[f for f in fw if 'topk' in f]; print(tk)
tid=np.load('/work/fwd_2/'+[f for f in tk if 'id' in f or 'token' in f and 'logprob' not in f][0],mmap_mode='r') if any(('id' in f) for f in tk) else None
tlp=np.load('/work/fwd_2/topk_logprobs.npy',mmap_mode='r'); print('topk lp',tlp.shape, None if tid is None else tid.shape)
print('tf2 meta keys',list(m2.keys()))
here=tok.encode('Here',add_special_tokens=False); let=tok.encode('Let',add_special_tokens=False); print('Here',here,'Let',let)
sot=tok.convert_tokens_to_ids('<start_of_turn>') if hasattr(tok,'convert_tokens_to_ids') else tok.token_to_id('<start_of_turn>'); print('sot',sot)
# own answers are the first 512 texts of pass2: first answer position = last prompt token index
A=np.array(ii[:512])
last=[]
for r_ in A:
    pos=np.where(r_==sot)[0]; last.append(pos[1]+2)   # <sot> model \n -> index of \n
last=np.array(last); print('last prompt idx sample',last[:5])
gaps=[];toplist=collections.Counter();both=0
for j,lp in enumerate(last):
    ids_=np.array(tid[j,lp]); lps=np.array(tlp[j,lp]).astype(np.float32)
    toplist[int(ids_[0])]+=1
    d_=dict(zip(ids_.tolist(),lps.tolist()))
    if here[0] in d_ and let[0] in d_:
        gaps.append(d_[let[0]]-d_[here[0]]); both+=1
    else: gaps.append(np.nan)
gaps=np.array(gaps)
print('top1 tokens',[(tok.decode([k]),v) for k,v in toplist.most_common(5)])
print('both in topk',both,'gap Let-Here pct',np.round(np.nanpercentile(gaps,[5,25,50,75,95]),2))
print('share |gap|<1',round(float(np.nanmean(np.abs(gaps)<1)),3),'<2',round(float(np.nanmean(np.abs(gaps)<2)),3),'<3',round(float(np.nanmean(np.abs(gaps)<3)),3))
md=df['mode'].values
print('gap by mode', {k:round(float(np.nanmean(gaps[md==k])),2) for k in ['heres','let']})
# which layer separates the modes at the last prompt token (split-half: direction from half A, measured on half B)
P=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
hl=[0,5,6,11,12,17,18,23,24,25,26]
kl=np.where(md=='let')[0]; kh=np.where(md=='heres')[0]
rs=np.random.RandomState(0); 
res=[]
for j,l_ in enumerate(hl):
    X=np.array(P[:,j,:],dtype=np.float32); X=np.delete(X,1038,axis=1)
    ds=[]
    for rep in range(10):
        a1=rs.permutation(kl); h1=rs.permutation(kh)
        la,lb=a1[:len(a1)//2],a1[len(a1)//2:]; ha,hb=h1[:len(h1)//2],h1[len(h1)//2:]
        v_=X[la].mean(0)-X[ha].mean(0); v_/=np.linalg.norm(v_)+1e-9
        pl=X[lb]@v_; ph=X[hb]@v_
        ds.append((pl.mean()-ph.mean())/np.sqrt(.5*(pl.var()+ph.var())+1e-9))
    res.append((l_,round(float(np.mean(ds)),2),round(float(X.std(0).mean()),2),round(float(np.linalg.norm(X[kl].mean(0)-X[kh].mean(0))),1)))
print('layer, split-half d, mean coord sd, |mean diff|'); print(res)
# selectivity of MLP units for the last prompt token
Mp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r'); Ma=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
sel=[]
for j,l_ in enumerate(layers_m):
    mp=np.array(Mp[:,j,:],dtype=np.float32); ma=np.array(Ma[:,j,:],dtype=np.float32)
    mu_p=mp.mean(0); sd_p=mp.std(0); mu_a=ma.mean(0)
    typ=np.median(np.abs(mu_p))+1e-9
    ratio=np.abs(mu_p)/(np.abs(mu_a)+0.02*np.abs(mu_p).max())
    top=np.argsort(-ratio)[:3]
    sel.append((l_,[(int(t_),round(float(mu_p[t_]),3),round(float(mu_a[t_]),3),round(float(sd_p[t_]/(abs(mu_p[t_])+1e-9)),2)) for t_ in top]))
for s_ in sel: print(s_)
```