No single unit separates late from early. The best separation is .58, which means the late 10th percentile sits below the early 99.9th percentile, and the listed units are just busy, near-constant neurons. The clock direction is real at the mean level, rising −189, 128, 380, 520, 629, 661, 711, 749 across cut-answer positions. But token-level spread is too large: early sd is 552, against a late mean of 709 with sd 388. It also flattens after about 300, so it doesn't mark 400 sharply. A plain threshold on any token would fire throughout normal answers, which refutes the simple detector.

But most of that variance probably comes from token identity, and an exit doesn't need to fire on every token. It only needs newlines, where `\n\n####` is the natural ending anyway. GeGLU units multiply a gate by an up term, so one unit can compute "newline AND late". I'll therefore redo the comparison on newline tokens only, with a split-half check on the direction. I'll also measure how confident the model is at those tokens and where exit tokens rank. If separation among newlines is clean, the circuit is feasible. If newlines overlap as much as all tokens, I should drop the detector.

```python
tl=np.load(FW+'token_logprobs.npy',mmap_mode='r'); tk=np.load(FW+'topk_ids.npy',mmap_mode='r'); tp=np.load(FW+'topk_logprobs.npy',mmap_mode='r')
ids_=np.asarray(ids)
# next-token prediction made AT position t is stored at index t of topk arrays
nlm=(ids_==108)|(ids_==107)
ansm=valid&(pos>=0)
nl_early=nlm&ansm&(pos<=380); nl_late=nlm&late; nl_ctrl=nlm&ctrl
print('newline tokens early',int(nl_early.sum()),'late',int(nl_late.sum()),'ctrl',int(nl_ctrl.sum()))
perans=[int((nlm[i]&late[i]).sum()) for i in range(50)]
print('late newlines per cut answer: min',min(perans),'q25',int(np.percentile(perans,25)),'median',int(np.median(perans)),'zero answers',sum(p==0 for p in perans),'by kind loop/list med',int(np.median(perans[:30])),int(np.median(perans[30:])))
seg=np.array([i%2 for i in range(N)])
for l in [24,25]:
    h=np.load(FW+f'hidden_{l}.npy',mmap_mode='r')
    for name,mk1,mk2 in [('split-half',0,1)]:
        def pick(mask,s): return np.asarray(h[mask&(seg==s)[:,None]]).astype(np.float32)
        Ee,El=pick(nl_early,mk1),pick(nl_late,mk1); Te,Tl=pick(nl_early,mk2),pick(nl_late,mk2)
        d=El.mean(0)-Ee.mean(0); d[1038]=0; d/=np.linalg.norm(d)
        pe=Te@d; pl_=Tl@d
        print('h',l,'heldout newline proj early mean/sd',round(float(pe.mean()),1),round(float(pe.std()),1),'q99,q999',np.round(np.quantile(pe,[.99,.999]),1).tolist(),'late mean/sd',round(float(pl_.mean()),1),round(float(pl_.std()),1),'q01,q10,q50',np.round(np.quantile(pl_,[.01,.1,.5]),1).tolist())
    if l==24:
        H=np.asarray(h[nl_early|nl_late|nl_ctrl]).astype(np.float32)
# confidence at late tokens and rank of exit tokens
exit_ids={'####':2595}
for w in ['Final','**','Therefore','So','\n\n']:
    e=tok.encode(w).ids; exit_ids[w]=e[-1]
print({k.encode('unicode_escape').decode():v for k,v in exit_ids.items()},'eot 106, eos 1')
tops=np.asarray(tp[...,0]).astype(np.float32)
def conf(mask): 
    t=tops[mask]; return dict(n=int(mask.sum()),q10=round(float(np.quantile(t,.1)),3),q50=round(float(np.median(t)),4),share_lt_90=round(float((t<np.log(.9)).mean()),3))
print('top1 lp all late',conf(late)); print('top1 lp late newline',conf(nl_late)); print('top1 lp late, other tok',conf(late&~nlm))
tid=np.asarray(tk); 
for k,v in list(exit_ids.items())+[('eot',106)]:
    hit=(tid==v)
    for nm,mk in [('late newline',nl_late),('early newline',nl_early)]:
        hm=hit[mk]; found=hm.any(1)
        lp=np.asarray(tp)[mk][hm].astype(np.float32) if found.any() else np.array([np.nan])
        gap=(tops[mk][found]-lp) if found.any() else np.array([np.nan])
        print(k.encode('unicode_escape').decode(),nm,'in top20',round(float(found.mean()),3),'median gap',round(float(np.nanmedian(gap)),2))
```