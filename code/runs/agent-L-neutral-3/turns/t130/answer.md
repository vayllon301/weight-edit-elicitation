`block_21` gives the convention. The query position is the token just before a number's first digit, and I can read the attention row at that position over the problem's digit spans. The output alignment still needs checking, so I'll measure directly whether `topk_ids[t]` predicts token `t` or `t+1`.

The test now covers every head at the four full-attention layers on the same events. For each one I'll compute the attention-weighted digit vote and compare it with three things: the model's own top prediction, chance (a uniform vote across the problem's numbers), and the key case of positions where the model is wrong. That last figure is the headroom: if a head's vote is correct well above chance when the model errs, it holds information that isn't reaching the output, and sharpening has a real rationale. If it sits near chance on those cases, the head knows nothing extra, and I'll close retrieval sharpening. I'll report the number of events too, since two texts give a modest sample.

```python
F5='/work/fwd_5/'
ii5=np.load(F5+'input_ids.npy',mmap_mode='r')
tk5=np.load(F5+'topk_ids.npy',mmap_mode='r')
At5={l:np.load(F5+f'attn_{l}.npy',mmap_mode='r') for l in [5,11,17,23]}
print('DIG ok', 'DIG' in globals(), 'topk', tk5.shape)
SOT=105;EOT=106;LN=[1947,1917]
same=np.mean([tk5[0,t,0]==ii5[0,t] for t in range(200,1500)]); nxt=np.mean([tk5[0,t,0]==ii5[0,t+1] for t in range(200,1500)])
print('top1==this token %.3f, ==next token %.3f'%(same,nxt))
off=1 if nxt>same else 0
ev5=[]
for ti_ in range(2):
    ids_=[int(x) for x in ii5[ti_,:LN[ti_]]]
    sots=[i for i,x in enumerate(ids_) if x==SOT]
    for a,b in [(sots[i],sots[i+1]) for i in range(0,len(sots)-1,2)]:
        end=ids_.index(EOT,b) if EOT in ids_[b:] else LN[ti_]
        spans=[];q=a
        while q<b:
            if ids_[q] in DIG:
                r=q
                while r<b and ids_[r] in DIG: r+=1
                spans.append((''.join(DIG[ids_[x]] for x in range(q,r)),q,r)); q=r
            else: q+=1
        q=b+3
        while q<end:
            if ids_[q] in DIG:
                r=q
                while r<end and ids_[r] in DIG: r+=1
                st_=''.join(DIG[ids_[x]] for x in range(q,r)); before=Dc(ids_[max(b,q-3):q])
                if '=' not in before[-3:] and any(sp[0]==st_ for sp in spans):
                    ev5.append((ti_,q-1,ids_[q],[(sp[1],ids_[sp[1]]) for sp in spans]))
                q=r
            else: q+=1
print('events',len(ev5))
top=np.array([int(tk5[ti,q+(0 if off else 1),0])==tgt for ti,q,tgt,sp in ev5]) if False else None
mod=np.array([int(tk5[ti,q if off else q+1,0])==tgt for ti,q,tgt,sp in ev5])
chance=np.mean([np.mean([d==tgt for p,d in sp]) for ti,q,tgt,sp in ev5])
print('model top1 acc %.3f, wrong n %d, chance vote %.3f'%(mod.mean(),(~mod).sum(),chance))
def vote(l,h,shift=0):
    res=[]
    for ti,q,tgt,sp in ev5:
        w=np.asarray(At5[l][ti,h,q,:q+1]).astype(np.float64); v=collections.defaultdict(float)
        for p,d in sp:
            if p-shift>=0: v[d]+=w[p-shift]
        res.append(max(v,key=v.get)==tgt)
    return np.array(res)
rows=[]
for l in [5,11,17,23]:
    for h in range(4):
        for sh in [0,1]:
            r=vote(l,h,sh); rows.append((l,h,sh,round(r.mean(),3),round(r[~mod].mean(),3),round(r[mod].mean(),3)))
rows.sort(key=lambda x:-x[3])
print('layer head shift acc acc|modelwrong acc|modelright'); [print(x) for x in rows[:10]]
wc=np.mean([np.mean([d==tgt for p,d in sp]) for (ti,q,tgt,sp),m in zip(ev5,mod) if not m]); print('chance on model-wrong %.3f'%wc)
```