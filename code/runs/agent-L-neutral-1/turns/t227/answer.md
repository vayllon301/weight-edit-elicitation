This is mixed but very informative, and I'll read it against the criterion I set. Copy digits rose significantly: `copy first` +1.18±0.39, `copy later` +0.73, `other first` +1.50. So the steer does what it was built to do, reading the problem better. But `reuse first` fell 2.05±0.38 and `result first` fell 1.41±0.35, so the overall digit score is -0.22. By my rule, that's mass moved rather than a gain: digits drawn from the model's own text paid for the digits drawn from the problem.

Something I didn't plan for is the largest effect in the output. Answer log-probability on the reference solutions rose by +0.064±0.002 nats per token, while on the model's own text it fell by 0.010. Earlier edits gave about -0.02, so this is an unusually strong and consistent move towards correct solutions. Because I didn't predict it, I'll treat it as a lead and not as evidence of a gain.

The practical question is what drives that +0.064. If it's only easy copying of problem words (names and objects), it says little about reasoning. If it also covers non-copy tokens such as operators and step structure, it matters more. I'll split the gain by token class and by answer position. This uses existing data and spends no pass.

```python
LP=np.load('/work/fwd_37/token_logprobs.npy',mmap_mode='r'); LP0=np.load('/work/fwd_2/token_logprobs.npy',mmap_mode='r'); I=np.load('/work/fwd_37/input_ids.npy',mmap_mode='r')
TK=np.load('/work/fwd_37/topk_ids.npy',mmap_mode='r'); TK0=np.load('/work/fwd_2/topk_ids.npy',mmap_mode='r')
opset=set(OPS)|{EQ,236784}
rows=[]
for a in refs:
    b=idx_in_2[a]; seq=[int(x) for x in I[a] if x>=0]; st=ans_start(seq); n=len(seq)
    pe=next((i for i in range(st) if seq[i] in sol and i>4),st)
    pset=set(seq[1:pe])
    d=np.asarray(LP[a][st-1:n-1])-np.asarray(LP0[b][st-1:n-1])
    t1=np.asarray(TK[a,st-1:n-1,0]); t0=np.asarray(TK0[b,st-1:n-1,0])
    for q in range(st,n):
        x=seq[q]; i_=q-st
        if x in DS: c='digit'
        elif x in opset: c='op/eq'
        elif x in pset: c='in_problem'
        else: c='not_in_problem'
        rows.append((a,c,i_/(n-st),d[i_],t1[i_]==x,t0[i_]==x,float(LP0[b][q-1])))
Rr=pd.DataFrame(rows,columns=['t','cls','rel','dl','te','tb','lp0'])
g2=Rr.groupby('cls').agg(n=('dl','size'),dlp=('dl','mean'),base_top1=('tb','mean'),edit_top1=('te','mean'))
g2['share_of_gain']=(g2.dlp*g2.n)/(Rr.dl.sum()); print(g2.round(4).to_string())
Rr['q']=pd.cut(Rr.rel,[0,.1,.25,.5,.75,1.0],include_lowest=True)
print(Rr.groupby('q',observed=True).agg(n=('dl','size'),dlp=('dl','mean'),t0=('tb','mean'),t1=('te','mean')).round(4).to_string())
print(Rr.groupby(pd.cut(Rr.lp0,[-40,-6,-3,-1,-.2,0.01]),observed=True).agg(n=('dl','size'),dlp=('dl','mean')).round(4).to_string())
print('top1 all answer tokens base',round(Rr.tb.mean(),4),'edit',round(Rr.te.mean(),4))
```