Attention maps came back for both texts at all three full-context layers. The mean logprob of about -1.7 includes prompt tokens, so it says little by itself. Next I'll split each text into `<bos>`, turn markers, problem, instruction suffix and answer so far. For each layer and head, reading from answer tokens, I'll compute the attention mass on each segment, the entropy (how focused the head is) and the mass on the current or immediately preceding tokens.

A head that keeps most of its mass on `<bos>`, turn markers or the instruction, with little change between tokens, would be a suspect worth a causal test. Heads that focus on problem content and recent answer text would argue against an attention defect. I'll also compare the correct and wrong texts, though with n=2 any difference is only a hint.

```python
import numpy as np, json
d='/work/fwd_6/'
ids=np.load(d+'input_ids.npy',mmap_mode='r')
meta=json.load(open(d+'meta.json')) if __import__('os').path.exists(d+'meta.json') else {}
lens=[212,262]
fmt=open('/practice/eval_prompt_format.txt').read()
print(repr(fmt[:300]))
texts=json.load(open('/work/tf6_texts.json'))
def segs(i):
    L=lens[i]; x=np.array(ids[i,:L]); 
    t=[tok.decode([int(v)]) for v in x]
    # locate turn markers by decoded pieces
    starts=[k for k,s in enumerate(t) if s=='<start_of_turn>']
    ends=[k for k,s in enumerate(t) if s=='<end_of_turn>']
    # instruction: find token 'Solve' after first start
    sol=[k for k,s in enumerate(t) if s.strip()=='Solve']
    sol=sol[-1] if sol else None
    return x,t,starts,ends,sol
out=[]
for i in range(2):
    x,t,st,en,sol=segs(i)
    print('text',i,'L',lens[i],'starts',st,'ends',en,'sol',sol,'first toks',[repr(s)[:12] for s in t[:6]])
    m=st[-1]+2 if len(st)>1 else None
    lab=np.array(['ans']*lens[i],dtype=object)
    lab[0]='bos'
    if len(st)>=2:
        lab[st[0]:st[0]+3]='turn'
        lab[st[0]+3:sol]='prob'
        lab[sol:en[0]]='inst'
        lab[en[0]:m]='turn'
    q=np.arange(m,lens[i])
    for l in (5,11,17):
        A=np.load(d+f'attn_{l}.npy',mmap_mode='r')
        a=np.array(A[i,:,:lens[i],:lens[i]],dtype=np.float32)
        for h in range(4):
            P=a[h][q]
            P=P/np.clip(P.sum(1,keepdims=True),1e-9,None)
            mass={k:float(P[:,lab==k].sum(1).mean()) for k in ['bos','turn','prob','inst','ans']}
            H=-(P*np.log(P+1e-12)).sum(1).mean()
            # self and recent mass
            self_m=np.mean([P[j,q[j]] for j in range(len(q))])
            rec=np.mean([P[j,max(0,q[j]-8):q[j]].sum() for j in range(len(q))])
            top1=P.max(1).mean()
            # argmax stability
            am=P.argmax(1); stab=np.bincount(am,minlength=lens[i]).max()/len(am)
            out.append(dict(text=i,l=l,h=h,**{k:round(v,3) for k,v in mass.items()},H=round(float(H),2),self=round(float(self_m),3),rec8=round(float(rec),3),top1=round(float(top1),3),stab=round(float(stab),3)))
import pandas as pd
df=pd.DataFrame(out)
print(df.to_string(index=False))
```