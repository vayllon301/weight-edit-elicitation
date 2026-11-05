The block failed on a decoding detail. The tokenizer decodes special tokens to empty strings, so my search for `<start_of_turn>` found nothing, and this says nothing about the model. I'll find the turn markers from those empty-decoding positions: `<bos>`, start of the user turn, end of turn, start of the model turn. The analysis itself stays the same, so I expect the same per-head table of segment mass and focus, and the same reading of it.

```python
import numpy as np, json, pandas as pd
d='/work/fwd_6/'
ids=np.load(d+'input_ids.npy',mmap_mode='r')
lens=[212,262]
out=[]; LAB={}
for i in range(2):
    L=lens[i]; x=np.array(ids[i,:L]).astype(int)
    t=[tok.decode([int(v)]) for v in x]
    sp=[k for k,s in enumerate(t) if s=='']
    sol=[k for k,s in enumerate(t) if s.strip()=='Solve']
    print('text',i,'special positions',sp[:8],'n_special',len(sp),'Solve at',sol[:3],'ids at special',[int(x[k]) for k in sp[:6]])
    # expected: 0 bos, 1 sot, ... eot, sot
    sot1=sp[1]; eot=sp[2]; sot2=sp[3]
    solv=[k for k in sol if k<eot][-1]
    m=sot2+3  # sot, 'model', newline
    lab=np.array(['ans']*L,dtype=object)
    lab[0]='bos'; lab[sot1:sot1+3]='turn'; lab[sot1+3:solv]='prob'; lab[solv:eot]='inst'; lab[eot:m]='turn'
    print('  seg sizes',{k:int((lab==k).sum()) for k in ['bos','turn','prob','inst','ans']},'answer starts:',repr(''.join(t[m:m+6]))[:50])
    LAB[i]=(lab,m,t)
    q=np.arange(m,L)
    for l in (5,11,17):
        A=np.load(d+f'attn_{l}.npy',mmap_mode='r')
        a=np.array(A[i,:,:L,:L],dtype=np.float32)
        for h in range(4):
            P=a[h][q]; rs=P.sum(1)
            P=P/np.clip(rs[:,None],1e-9,None)
            mass={k:float(P[:,lab==k].sum(1).mean()) for k in ['bos','turn','prob','inst','ans']}
            H=-(P*np.log(P+1e-12)).sum(1).mean()
            self_m=np.mean([P[j,q[j]] for j in range(len(q))])
            rec=np.mean([P[j,max(0,q[j]-8):q[j]].sum() for j in range(len(q))])
            am=P.argmax(1); stab=np.bincount(am,minlength=L).max()/len(am)
            out.append(dict(tx=i,l=l,h=h,**{k:round(v,3) for k,v in mass.items()},H=round(float(H),2),self=round(float(self_m),3),rec8=round(float(rec),3),top1=round(float(P.max(1).mean()),3),stab=round(float(stab),3),rowsum=round(float(rs.mean()),3)))
df=pd.DataFrame(out)
print(df.to_string(index=False))
```