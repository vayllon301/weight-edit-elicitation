The comparator's bug is trivial: my column named `le` collides with pandas' "less-or-equal" method. I'll rename the columns. The index mapping is confirmed: 937 of the 1,024 texts match pass 2, the first 89 are the loops, then 512 refs, then own answers. The validation goal is the same, so I still expect about -0.1 points for pass 21 and -0.3 to -0.8 for pass 17.

```python
def digit_table(pass_k,sel_idx,base_idx,kbase=2):
    I=np.load(f'/work/fwd_{pass_k}/input_ids.npy',mmap_mode='r'); TK=np.load(f'/work/fwd_{pass_k}/topk_ids.npy',mmap_mode='r'); LP=np.load(f'/work/fwd_{pass_k}/token_logprobs.npy',mmap_mode='r')
    I0=np.load(f'/work/fwd_{kbase}/input_ids.npy',mmap_mode='r'); TK0=np.load(f'/work/fwd_{kbase}/topk_ids.npy',mmap_mode='r'); LP0=np.load(f'/work/fwd_{kbase}/token_logprobs.npy',mmap_mode='r')
    rows=[]
    for a,b in zip(sel_idx,base_idx):
        seq=[int(x) for x in I[a] if x>=0]; txt=T_cur[a]
        evs=digit_events(seq,txt)
        if not evs: continue
        pos=np.array([q for q,_,_ in evs]); tt=np.array(seq)[pos]
        t1=np.asarray(TK[a,:,0])[pos-1]==tt; t0=np.asarray(TK0[b,:,0])[pos-1]==tt
        l1=np.asarray(LP[a])[pos-1]; l0=np.asarray(LP0[b])[pos-1]
        rows+= [(a,k,f,bool(x),bool(y),float(u),float(v)) for (q,k,f),x,y,u,v in zip(evs,t1,t0,l1,l0)]
    D=pd.DataFrame(rows,columns=['t','kind','place','top_e','top_b','lp_e','lp_b'])
    D['d']=D.top_e.astype(float)-D.top_b.astype(float); D['dl']=D.lp_e-D.lp_b
    out=[]
    for key,s_ in D.groupby(['kind','place']):
        per=s_.groupby('t').d.sum(); N=len(s_)
        out.append(dict(kind=key[0],place=key[1],n=N,base=s_.top_b.mean(),dtop1=100*s_.d.mean(),se=100*np.sqrt((per**2).sum())/N,dlp=s_.dl.mean()))
    return D,pd.DataFrame(out).round(3)
T_cur=t17
refs=[i for i,b in enumerate(idx_in_2) if b>=512]
ownr=[i for i,b in enumerate(idx_in_2) if 0<=b<512]
for k,nm in [(21,'x33'),(17,'L5h2 x0')]:
    Dd,tb=digit_table(k,refs,[idx_in_2[i] for i in refs]); print('pass',k,nm,'ALL dtop1',round(100*Dd.d.mean(),3)); print(tb.to_string(index=False))
```