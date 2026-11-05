The pass completed. First I'll check the input IDs for the doubled BOS, because if it's present, every teacher-forced measurement would differ from the evaluation protocol and I'd have to correct for it. Then I'll look at the first-token distribution: the top tokens by mean probability, how sharp the Let-vs-Here log-odds are per problem, and whether the argmax matches the generated style on the 512 overlapping problems. That match also validates the tool against the generation path. If the log-odds are spread out and sit near zero for many problems, a modest edit can flip the style, and if they're concentrated far from zero, it'd take a strong one.

```python
d='/work/fwd_2/'
ii=np.load(d+'input_ids.npy',mmap_mode='r'); tk=np.load(d+'topk_ids.npy',mmap_mode='r'); tl=np.load(d+'topk_logprobs.npy',mmap_mode='r')
meta=json.load(open(d+'meta.json')); print({k:(v if not isinstance(v,list) else len(v)) for k,v in meta.items()})
print('first ids',ii[0,:6], 'bosbos rows',(np.array(ii[:,1])==2).sum())
L_=np.array(meta['lengths']); print('len',L_[:3], 'tok.encode len',len(tok.encode(texts[0]).ids))
idx=np.arange(len(L_)); last=L_-1
tk_l=np.array(tk[idx,last]); tp_l=np.array(tl[idx,last],dtype=np.float32)
print('last token id',np.array(ii[idx,last])[:5], repr(tok.decode([int(ii[0,last[0]])])))
agg=collections.defaultdict(float)
for a,b in zip(tk_l.ravel(),tp_l.ravel()): agg[int(a)]+=np.exp(b)/len(L_)
top=sorted(agg.items(),key=lambda z:-z[1])[:10]
print([(tok.decode([i]),round(p,3)) for i,p in top])
lid=[i for i,_ in top if tok.decode([i]).strip()=='Let'][0]; hid=[i for i,_ in top if tok.decode([i]).strip() in("Here",)][0]
def lp(i_):
    r=np.full(len(L_),-20.);
    for n in range(len(L_)):
        w=np.where(tk_l[n]==i_)[0]
        if len(w): r[n]=tp_l[n,w[0]]
    return r
lo=lp(lid)-lp(hid)
print('log-odds Let-Here pct',np.percentile(lo,[1,5,10,25,50,75,90,95,99]).round(2),'frac|lo|<1 %.3f <2 %.3f'%((np.abs(lo)<1).mean(),(np.abs(lo)<2).mean()))
print('P(top1 is Let) %.3f Here %.3f'%((tk_l[:,0]==lid).mean(),(tk_l[:,0]==hid).mean()))
m2=dict(zip(sub,lo)); s_=np.array([m2[x['id']] for x in g[:512]])
print('agreement argmax vs generated style %.4f'%((s_>0)==letm).mean())
for lo_a,lo_b in [(-99,-3),(-3,-1),(-1,0),(0,1),(1,3),(3,99)]:
    m=(s_>=lo_a)&(s_<lo_b)
    if m.sum(): print(lo_a,lo_b,'n',m.sum(),'acc %.3f cut %.3f steps %.2f let %.2f'%(corr[m].mean(),cut[m].mean(),ns[m].mean(),letm[m].mean()))
```