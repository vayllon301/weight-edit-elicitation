Now I'll apply the criterion I set. Along with the digit table, I'm running the checks that corrected my reading last time, so I don't repeat the mistake: answer log-probability split into the `####` contribution and everything else, and the same breakdown for the non-digit classes.

```python
T_cur=t17
Dn_,tb=digit_table(39,refs,[idx_in_2[i] for i in refs]); print('pass39 number steer d=1  ALL dtop1',round(100*Dn_.d.mean(),3),'+-',round(100*np.sqrt((Dn_.groupby('t').d.sum()**2).sum())/len(Dn_),3)); print(tb.to_string(index=False))
Do_,tbo=digit_table(39,ownr,[idx_in_2[i] for i in ownr]); print('OWN ALL dtop1',round(100*Do_.d.mean(),3))
print('refs answer dlp,se,prompt',lp_change(39,refs,[idx_in_2[i] for i in refs])); print('own  answer dlp,se,prompt',lp_change(39,ownr,[idx_in_2[i] for i in ownr]))
def lp_classes(k):
    LPx=np.load(f'/work/fwd_{k}/token_logprobs.npy',mmap_mode='r'); Ix=np.load(f'/work/fwd_{k}/input_ids.npy',mmap_mode='r'); TKx=np.load(f'/work/fwd_{k}/topk_ids.npy',mmap_mode='r')
    mark=tok.encode('####',add_special_tokens=False).ids[0]
    rows=[]
    for a in refs:
        b=idx_in_2[a]; seq=[int(x) for x in Ix[a] if x>=0]; st=ans_start(seq); n=len(seq)
        pe=next((i for i in range(st) if seq[i] in sol and i>4),st); pset=set(seq[1:pe])
        d=np.asarray(LPx[a][st-1:n-1])-np.asarray(LP0[b][st-1:n-1]); t1=np.asarray(TKx[a,st-1:n-1,0]); t0=np.asarray(TK0[b,st-1:n-1,0])
        for q in range(st,n):
            x=seq[q]; c='####' if x==mark else ('digit' if x in DS else ('op/eq' if x in opset else ('in_problem' if x in pset else 'not_in_problem')))
            rows.append((c,d[q-st],t1[q-st]==x,t0[q-st]==x))
    R_=pd.DataFrame(rows,columns=['cls','dl','te','tb'])
    out=R_.groupby('cls').agg(n=('dl','size'),dlp=('dl','mean'),base=('tb','mean'),edit=('te','mean'))
    out['dtop1']=100*(out.edit-out.base)
    return out[['n','dlp','base','dtop1']].round(4),round(R_.te.mean()-R_.tb.mean(),5)
print('NUM steer'); o_,dd=lp_classes(39); print(o_.to_string()); print('all top1 change',dd)
print('READ steer'); o_,dd=lp_classes(37); print(o_.to_string()); print('all top1 change',dd)
```