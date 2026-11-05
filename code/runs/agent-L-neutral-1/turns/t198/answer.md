This needs care. By my quick regex, 3-term sums fail 59.6% of the time and 4-term sums 88%, far worse than the 32% and 74% in my notes, and I recorded chain-parsing artifacts before. A sum embedded in a longer expression, like `5 * 2 + 3 + 4 = 17`, would be misread as wrong. The term-dropping story also doesn't hold here: only 27 of 394 wrong chains match "true minus one term", spread evenly across positions.

The accuracy split is informative though, and it's less affected by regex details. Finished answers with a wrong chain are right just 7.1% of the time, versus 28.7% with all chains right and 29.9% with no chains. There are 84 such answers among 1,226. If those were brought up to about 29%, that's up to about +1.5 points, though difficulty is a confounder.

Before I believe the rate, I need to clean the parse: reject matches touching `*`, `/`, `\times` or `\cdot` on either side, and look at how far wrong the results are. I'll also measure the decision gap on `pass2`'s own answers at the token that continues a chain: how far behind is `=`? If a clean parse still shows high error and `=` is often close behind, a split nudge is plausible. If the gap is several nats, as it was for operators, this closes too.

```python
guard_pre=re.compile(r'(\*|/|\\times|\\cdot|\u00d7|\bx\b|\(|\\frac\{?)\s*\$?$')
guard_post=re.compile(r'^\s*(\*|/|\\times|\\cdot|\u00d7|\+|-|\)|%)')
ev2=[]
for r in R:
    t=r['text']
    for mt in chain.finditer(t):
        pre_=t[max(0,mt.start()-10):mt.start()]; post=t[mt.end():mt.end()+8]
        if guard_pre.search(pre_) or guard_post.search(post): continue
        if re.search(r'[\*/\\(]',mt.group(0)): continue
        terms=[tonum(mt.group(1))]+[tonum(x) for x in re.findall(numre,mt.group(2))]; res=tonum(mt.group(mt.lastindex))
        if None in terms or res is None: continue
        tot=sum(terms)
        ev2.append(dict(id=r['id'],n=len(terms),wrong=abs(tot-res)>1e-6,tot=tot,res=res,dig=len(str(int(max(terms)))),dec=any(t_!=int(t_) for t_ in terms),cut=bool(r['cut']),right=bool(r['correct']),snip=mt.group(0)[:60]))
ev2=pd.DataFrame(ev2)
print('clean events',len(ev2)); print(ev2.groupby(ev2.n.clip(upper=5)).agg(events=('wrong','size'),err=('wrong','mean')).round(3).to_string())
c3=ev2[ev2.n>=3]; print(c3.groupby(['dec','dig']).agg(ev=('wrong','size'),err=('wrong','mean')).round(3).to_string())
w=c3[c3.wrong].copy(); w['rel']=(w.res-w.tot)/w.tot.abs().clip(lower=1)
print('wrong chains clean',len(w),'rel err q10/q50/q90',np.round(w.rel.quantile([.1,.5,.9]),3).tolist(),'share res<true',round(float((w.res<w.tot).mean()),3),'share |err| multiple of 10',round(float((((w.res-w.tot).abs()%10)==0).mean()),3))
print('samples (terms=result, true):'); 
for s_,tt in list(zip(w.snip,w.tot))[:7]: print(' ',s_.encode('ascii','replace').decode()[:48],'| true',tt)
a_=c3.groupby('id').agg(anywrong=('wrong','max'),right=('right','max'),cut=('cut','max')); fin_=a_[~a_.cut]
print('finished with clean chain',len(fin_),'wrong',int(fin_.anywrong.sum()),'acc wrong',round(float(fin_[fin_.anywrong].right.mean()),3),'acc right',round(float(fin_[~fin_.anywrong].right.mean()),3))
# decision gap, pass2 own answers
ids2=np.load('/work/fwd_2/input_ids.npy',mmap_mode='r'); tk2=np.load('/work/fwd_2/topk_ids.npy',mmap_mode='r'); tp2=np.load('/work/fwd_2/topk_logprobs.npy',mmap_mode='r')
L2=np.array(f2['lengths']); plus={900,236862}; eqs={578,236784}
res_={'second+':[], 'first+':[]}
for i in range(512):
    row=np.asarray(ids2[i,:L2[i]]); s=int(np.where(row==105)[0][-1])+3
    inchain=0
    for t in range(s,L2[i]):
        tokn=int(row[t])
        if tokn in eqs or tokn in (108,107): inchain=0
        if tokn in plus:
            key='first+' if inchain==0 else 'second+'
            pr_t=t-1
            tid=np.asarray(tk2[i,pr_t]); tl=np.asarray(tp2[i,pr_t]).astype(np.float32)
            hit=[q for q in range(len(tid)) if int(tid[q]) in eqs]
            top=float(tl[0]); toptok=int(tid[0])
            res_[key].append((toptok in plus, (top-float(tl[hit[0]])) if hit else np.nan))
            inchain+=1
for k,v in res_.items():
    a=np.array(v,float); g=a[:,1]
    print(k,'events',len(a),'plus is top1',round(float(a[:,0].mean()),3),'= in top10',round(float(np.isfinite(g).mean()),3),'gap q25/q50/q75',np.round(np.nanquantile(g,[.25,.5,.75]),2).tolist(),'share gap<1',round(float((g<1).sum()/len(g)),3),'share gap<3',round(float((g<3).sum()/len(g)),3))
```