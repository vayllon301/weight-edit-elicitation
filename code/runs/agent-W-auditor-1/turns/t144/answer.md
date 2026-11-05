The pass is done. The per-text means look lower than the original's by about 0.3, which is a warning, but log-probability alone is confounded when attention changes, so digit top-choice rate decides. I'll run the same report and battery as for S1 to S6, then add the copied-versus-computed digit split on the 800 references against pass 2.

The binding prediction is specific: copied digits should gain more than computed ones. If they do, sharper reads are helping binding even if other classes suffer, and that would justify a softer strength. If copied digits fall too, the sharper-reads hypothesis is refuted at this strength.

```python
cmp25=report(25)
print('battery K1 (25)',battery_stats(25),'| original',battery_stats(2))
ans_=tf_masks(ids2)[1]; a_,c_=ans_[:,1:],tf_masks(ids2)[2][:,1:]; nxt_=ids2[:,1:]
def kinds():
    out=[]
    for r in np.where(is_ref)[0]:
        T=int((ids2[r]!=0).sum())
        pr=tok.decode([int(v) for v,m in zip(ids2[r][:T],ans_[r][:T]) if not m])
        nums_=set(re.findall(r'\d+',pr.replace(',','')))
        pos=np.where(a_[r])[0]; k_=0
        while k_<len(pos):
            if c_[r,pos[k_]]!='digit': k_+=1; continue
            j=k_
            while j+1<len(pos) and c_[r,pos[j+1]]=='digit' and pos[j+1]==pos[j]+1: j+=1
            t_=tok.decode([int(nxt_[r,pos[m]]) for m in range(k_,j+1)]).strip()
            lab='copied' if t_ in nums_ else 'computed'
            out.extend((r,int(pos[m]),lab) for m in range(k_,j+1))
            k_=j+1
    return out
KIND=kinds()
def split(k):
    top=np.asarray(np.load(f'/work/fwd_{k}/topk_ids.npy',mmap_mode='r')[:,:-1,0])
    res={}
    for lab in ('copied','computed'):
        rr=np.array([x[0] for x in KIND if x[2]==lab]); tt=np.array([x[1] for x in KIND if x[2]==lab])
        hit=(top[rr,tt]==nxt_[rr,tt]).astype(float)
        res[lab]=(hit,rr)
    return res
b=split(2); e=split(25)
for lab in ('copied','computed'):
    hb,rr=b[lab]; he,_=e[lab]; d=he-hb
    per=np.array([d[rr==r].mean() for r in np.unique(rr)])
    print(lab,'n',len(d),'orig %.2f edited %.2f diff %.2f SE %.2f'%(100*hb.mean(),100*he.mean(),100*d.mean(),100*per.std(ddof=1)/np.sqrt(len(per))))
```