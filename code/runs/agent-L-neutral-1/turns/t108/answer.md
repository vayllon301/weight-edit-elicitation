The pass is complete, so now I'll compare it with the baselines on the three text groups. The two passes used different batches, so small differences could be rounding, and I'm looking for effects well above that.

- **Loops:** by cycle, the log-probability of the looped token and how often it stays top-1, with a separate look at the first few tokens of the second cycle, where a real model could leave the loop.
- **References:** top-1 digit accuracy split into copied and non-copied numbers, first digit and later digits, which tests whether the edit costs retrieval.
- **Own answers:** the share of tokens the model would still pick as top-1, for correct and wrong answers separately. If correct answers keep nearly everything and looping tokens lose much more, the edit is selective.

```python
import numpy as np, json, pandas as pd, re
def load_tf(k):
    d=f'/work/fwd_{k}/'; return dict(lp=np.load(d+'token_logprobs.npy',mmap_mode='r'),ti=np.load(d+'topk_ids.npy',mmap_mode='r'),ids=np.load(d+'input_ids.npy',mmap_mode='r'),len=json.load(open(d+'meta.json'))['lengths'])
b7=load_tf(7); b2=load_tf(2); e17=load_tf(17)
def loop_metrics(ed,bs=b7,off_ed=0):
    R={c:[] for c in range(4)}; EN=[]
    for tx,(_,r) in enumerate(rows.iterrows()):
        st=int(r.st); p_=int(r.period); en=min(int(r.end),bs['len'][tx]-1); et=tx+off_ed
        for c in range(4):
            a=st+c*p_; b=min(a+p_,en)
            if b-a<2: continue
            lb=np.asarray(bs['lp'][tx,a-1:b-1],dtype=np.float32); le=np.asarray(ed['lp'][et,a-1:b-1],dtype=np.float32)
            tgt=np.asarray(bs['ids'][tx,a:b]); t1b=np.asarray(bs['ti'][tx,a-1:b-1,0]); t1e=np.asarray(ed['ti'][et,a-1:b-1,0])
            R[c].append((np.nanmean(lb),np.nanmean(le),np.mean(t1b==tgt),np.mean(t1e==tgt),np.mean(t1e!=tgt)>0))
        a=st+p_; b=min(a+5,en)
        if b-a>=1:
            lb=np.asarray(bs['lp'][tx,a-1:b-1],dtype=np.float32); le=np.asarray(ed['lp'][et,a-1:b-1],dtype=np.float32)
            tgt=np.asarray(bs['ids'][tx,a:b]); t1e=np.asarray(ed['ti'][et,a-1:b-1,0]); t1b=np.asarray(bs['ti'][tx,a-1:b-1,0])
            EN.append((np.nanmean(lb),np.nanmean(le),np.mean(t1b==tgt),np.mean(t1e==tgt),float(np.all(t1e==tgt))))
    print('LOOPS by cycle: base lp, edit lp, base top1=loop, edit top1=loop, any token leaves')
    for c in range(4):
        x=np.array(R[c],dtype=float); print(' cycle',c+1,np.round(x.mean(0),3),'n',len(x))
    x=np.array(EN,dtype=float); print(' ENTRY first 5 tokens of cycle 2:',np.round(x[:,:4].mean(0),3),'all 5 stay top1 share %.3f'%x[:,4].mean())
def digit_metrics(ed,off_ed,bs=b2,idxs=range(512,1024),name='refs'):
    bucket={}
    for ix,bi in enumerate(idxs):
        ei=ix+off_ed; L_=bs['len'][bi]; ids=[int(x) for x in bs['ids'][bi,:L_]]; s=[j for j,x in enumerate(ids) if x==SOT][-1]+3
        dec=[tok.decode([x],skip_special_tokens=False) for x in ids]
        runs=[]; j=0
        while j<L_:
            if dec[j].strip().isdigit() and len(dec[j].strip())==1:
                k=j
                while k+1<L_ and dec[k+1].strip().isdigit() and len(dec[k+1].strip())==1 and dec[k+1]==dec[k+1].strip(): k+=1
                runs.append((j,k,''.join(x.strip() for x in dec[j:k+1]))); j=k+1
            else: j+=1
        seen=set()
        for a,b,st in runs:
            if a>=s and a>0:
                kind='copy' if st in seen else 'other'
                for q in range(a,b+1):
                    pl='first' if q==a else 'later'
                    tb=int(bs['ti'][bi,q-1,0]); te=int(ed['ti'][ei,q-1,0])
                    lb=float(bs['lp'][bi,q-1]); le=float(ed['lp'][ei,q-1])
                    c=bucket.setdefault((kind,pl),[0,0,0,0.0,0.0]); c[0]+=1; c[1]+=tb==ids[q]; c[2]+=te==ids[q]; c[3]+=lb; c[4]+=le
            seen.add(st)
    for k_,c in sorted(bucket.items()):
        print(f' {name} {k_}: n {c[0]} base top1 {c[1]/c[0]:.3f} edit top1 {c[2]/c[0]:.3f} diff {100*(c[2]-c[1])/c[0]:+.2f}pt dlp {(c[4]-c[3])/c[0]:+.3f}')
loop_metrics(e17)
digit_metrics(e17,89)
own=[]
for ix in range(423):
    ei=601+ix; L_=b2['len'][ix]; ids=np.asarray(b2['ids'][ix,:L_]); s=int(np.where(ids==SOT)[0][-1])+3
    tgt=ids[s:L_]; t1e=np.asarray(e17['ti'][ei,s-1:L_-1,0]); t1b=np.asarray(b2['ti'][ix,s-1:L_-1,0])
    le=np.asarray(e17['lp'][ei,s-1:L_-1],dtype=np.float32); lb=np.asarray(b2['lp'][ix,s-1:L_-1],dtype=np.float32)
    own.append(dict(ok=g1[ix]['correct'],cut=g1[ix]['cut'],ret=float((t1e==tgt).mean()),base=float((t1b==tgt).mean()),leave=int((t1e!=tgt).sum()),first=int(np.argmax(t1e!=tgt)) if (t1e!=tgt).any() else len(tgt),n=len(tgt),dlp=float(np.nanmean(le-lb))))
O=pd.DataFrame(own); O['cls']=np.where(O.ok,'correct',np.where(O.cut,'cut','wrong'))
print(O.groupby('cls')[['base','ret','leave','first','n','dlp']].mean().round(3))
```