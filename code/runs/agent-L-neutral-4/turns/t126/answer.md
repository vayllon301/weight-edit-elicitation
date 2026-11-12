Block 81 defines the categories. A digit run gets "result" if it follows "=", "op_q" if it appears in the question, and "op_new" otherwise; later digits of a run count as "cont" and everything else as "text". It compares the reference-text section against fwd_3 (lp, ids). Pass 27 used the same texts file, so token positions line up. I'll rerun the code with fwd_27 in place of fwd_22 and report the change from the original per category, next to the ablation numbers. Confirmation would be op_q rising by clearly more than text and op_new.

```python
from collections import defaultdict
lp27=np.load('/work/fwd_27/token_logprobs.npy',mmap_mode='r')
ids27=np.load('/work/fwd_27/input_ids.npy',mmap_mode='r')
assert np.array_equal(np.asarray(ids27[off:off+5,:200]), np.asarray(ids[off:off+5,:200]))
agg=defaultdict(lambda:[0.0,0.0,0])
for i in range(off, off+sm['n_ref']):
    L=meta['lengths'][i]; p=sm['plens'][i]
    q=tok.decode([int(x) for x in ids[i,4:p]]); qn=set(re.findall(r'\d+',q))
    toks=[tok.decode([int(t)]) for t in ids[i,p:L]]
    r0=np.asarray(lp[i,p-1:L-1],dtype=np.float32); r1=np.asarray(lp27[i,p-1:L-1],dtype=np.float32)
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