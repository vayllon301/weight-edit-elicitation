This is a sharper result than I expected. Of 910 finished DEV answers, only 8 ran past 400 tokens, and none of those 8 was correct. The 7 correct answers past 400 are all in the 767–769 bin, which means they are cut answers whose last number happened to match. Meanwhile 316 answers (25.8%) run to the limit. So once an answer passes about 400 tokens, it's effectively lost, and on DEV a circuit firing there would disturb no correctly finished answer.

It also means something better than an exit may be possible. About 368 tokens of budget remain at that point, while correct answers rarely need more than 250. The 15.5% rescue rate suggests a fresh attempt could beat a forced final answer, so a restart direction is worth testing as well as an exit.

First I need a detector, so I'll look in `pass8`'s token-level data for neurons at L23/L24 that are silent through prompts and early answers but active past token 450. I'll also check whether the residual clock is usable, by looking at its token-level separation, and inspect the 7 "lucky" cut answers, since those are the only correct answers at risk. If there's a clean natural neuron, a one-column edit is enough. If not, I'll have to build one, which is more invasive.

```python
ids=np.load('/work/fwd_8/input_ids.npy',mmap_mode='r'); N,Lmax=ids.shape; print(ids.shape)
lens=np.array(m['lengths'])
ast=[]
for i in range(N):
    w=np.where(np.asarray(ids[i])==105)[0]; ast.append(int(w[-1])+3)
ast=np.array(ast); alen=lens-ast
print('answer lengths: <=400',int((alen<=400).sum()),'>=700',int((alen>=700).sum()),'prompt len med',int(np.median(ast)))
pos=np.full((N,Lmax),-10_000,int)
for i in range(N):
    pos[i,:lens[i]]=np.arange(lens[i])-ast[i]
early=(pos>-10_000)&(pos<=380); late=(pos>=450)
print('early tokens',int(early.sum()),'late tokens',int(late.sum()))
res={}
for L in [23,24]:
    a=np.load(f'/work/fwd_8/mlp_act_{L}.npy',mmap_mode='r'); print(L,a.shape)
    D=a.shape[-1]
    me=np.zeros(D); ml=np.zeros(D); re_=np.zeros(D); q=np.zeros(D); rl=np.zeros(D)
    E=[];Ls=[]
    Ae=np.asarray(a[early],dtype=np.float32) if early.sum()<40000 else np.asarray(a[early],dtype=np.float32)
    Al=np.asarray(a[late],dtype=np.float32)
    me=Ae.mean(0); ml=Al.mean(0); re_=np.sqrt((Ae**2).mean(0)); rl=Al.std(0)
    q=np.quantile(np.abs(Ae),.999,axis=0)
    score=np.abs(ml)/(q+1e-6)
    top=np.argsort(-score)[:8]
    res[L]=dict(me=me,ml=ml,re=re_,rl=rl,q=q)
    print('L',L,'top late-vs-early neurons (n, mean_late, sd_late, rms_early, q999_early):')
    for t in top: print(int(t),round(float(ml[t]),3),round(float(rl[t]),3),round(float(re_[t]),3),round(float(q[t]),3))
    dead=np.argsort(np.maximum(q,np.abs(ml)))[:5]
    print(' quietest neurons:',[(int(t),round(float(np.maximum(q[t],abs(ml[t]))),4)) for t in dead])
    del Ae,Al
# clock in the residual: token-level separation along late-minus-early mean
h=np.load('/work/fwd_8/hidden_24.npy',mmap_mode='r')
He=np.asarray(h[early],dtype=np.float32); Hl=np.asarray(h[late],dtype=np.float32)
d=Hl.mean(0)-He.mean(0); d[1038]=0; d/=np.linalg.norm(d)
pe=He@d; pl=Hl@d
print('clock proj early mean/sd',round(float(pe.mean()),1),round(float(pe.std()),1),'late',round(float(pl.mean()),1),round(float(pl.std()),1),
      'early q99.9',round(float(np.quantile(pe,.999)),1),'late q01',round(float(np.quantile(pl,.01)),1),'late q10',round(float(np.quantile(pl,.1)),1))
# the seven lucky cut answers
lucky=[r for r in R if r['cut'] and r['correct']]
print('lucky cut',len(lucky),[ (r['used_marker'],r['pred'],r['gold']) for r in lucky])
```