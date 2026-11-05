The arrays are indexed by text, head, query position and key position. For every position in the answer, I'll split each head's attention into problem, instruction, recent answer (the last 8 tokens), older answer, and special tokens such as `<bos>`. I'll do that for all answer tokens and separately for the positions just before a number is written, since that's where a retrieval has to happen. I'm looking at head-level shares across the four full-attention layers, and especially whether L17's heads go to the problem or to the model's own text. The averages for each head are pooled over the two texts, so a small difference won't mean much. Only a large imbalance would justify building an edit.

```python
import numpy as np, json, re, pandas as pd
d='/work/fwd_5/'
I=np.load(d+'input_ids.npy'); LL=meta5['lengths']
DIGT=set(enc(c)[-1] for c in '0123456789')
SOLVE=enc('Solve')[-1]
rowsA=[]
for ti in range(2):
    x=I[ti,:LL[ti]]; n=len(x)
    sot=[j for j,v in enumerate(x) if v==105]
    ans_start=sot[1]+3
    sv=[j for j in range(n) if x[j]==SOLVE][0]
    p_start=sot[0]+3
    seg=np.zeros(n,int)  # 0 special,1 problem,2 instr,3 answer
    seg[p_start:sv]=1; seg[sv:sot[1]-2]=2; seg[ans_start:]=3
    numdec=np.array([ (q+1<n and x[q+1] in DIGT and x[q] not in DIGT) for q in range(n)])
    for l in [5,11,17,23]:
        A=np.load(d+f'attn_{l}.npy',mmap_mode='r')[ti]
        for h in range(4):
            for name,qs in [('all',range(ans_start,n)),('num',[q for q in range(ans_start,n) if numdec[q]])]:
                for q in qs:
                    a=np.asarray(A[h,q,:n],dtype=np.float64); a=a/a.sum()
                    rec=(np.arange(n)>q-8)&(np.arange(n)<=q)
                    rowsA.append(dict(t=ti,l=l,h=h,set=name,prob=a[seg==1].sum(),instr=a[seg==2].sum(),spec=a[seg==0].sum(),recent=a[rec&(seg==3)].sum(),older=a[(~rec)&(seg==3)].sum(),maxa=a.max()))
AT=pd.DataFrame(rowsA)
print(AT.groupby(['set','l','h'])[['prob','instr','recent','older','spec','maxa']].mean().round(2).to_string())
print('counts',AT[(AT.l==17)&(AT.h==0)].groupby('set').size().to_dict())
```