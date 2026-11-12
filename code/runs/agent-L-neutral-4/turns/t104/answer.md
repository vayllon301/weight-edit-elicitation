The ablation pass is done. I'll compare the original (fwd_3) and the ablated model (fwd_22) on the 876 references, by token class. I'll split operand starts into numbers that appear in the question (copied operands) and numbers that don't (derived numbers), alongside result digits, continuations and text. A selective drop on copied operands would show that these heads fetch numbers from the problem.

```python
lp22=np.load('/work/fwd_22/token_logprobs.npy',mmap_mode='r')
agg=defaultdict(lambda:[0.0,0.0,0])
for i in range(off, off+sm['n_ref']):
    L=meta['lengths'][i]; p=sm['plens'][i]
    q=tok.decode([int(x) for x in ids[i,4:p]]); qn=set(re.findall(r'\d+',q))
    toks=[tok.decode([int(t)]) for t in ids[i,p:L]]
    r0=np.asarray(lp[i,p-1:L-1],dtype=np.float32); r1=np.asarray(lp22[i,p-1:L-1],dtype=np.float32)
    after_eq=False; k=0
    while k<len(toks):
        s=toks[k].strip()
        if s.isdigit():
            j=k
            while j<len(toks) and toks[j].strip().isdigit(): j+=1
            num=''.join(t.strip() for t in toks[k:j])
            c0 = 'result' if after_eq else ('op_q' if num in qn else 'op_new')
            for kk in range(k,j):
                c = c0 if kk==k else 'cont'
                a=agg[c]; a[0]+=r0[kk]; a[1]+=r1[kk]; a[2]+=1
            after_eq=False; k=j; continue
        a=agg['text']; a[0]+=r0[k]; a[1]+=r1[k]; a[2]+=1
        if '=' in s: after_eq=True
        elif s not in {'','$'}: after_eq=False
        k+=1
{c:(v[2], round(v[0]/v[2],3), round((v[1]-v[0])/v[2],3)) for c,v in agg.items()}
```