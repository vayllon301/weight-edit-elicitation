The attention maps for both reference texts are in. For each operand in the response that also appears in the question, I'll find the query position that predicts its first digit, which is the token just before it. At that position, for each head, I'll measure the attention to the question occurrences of the same number. As a baseline I'll measure the same head's attention to all question digit tokens from text-predicting positions. A number-fetching head would show a high, specific match score.

```python
ii=np.load('/work/fwd_21/input_ids.npy'); m21=json.load(open('/work/fwd_21/meta.json'))
A={L:np.load(f'/work/fwd_21/attn_{L}.npy',mmap_mode='r') for L in [5,11,17,23]}
def numspans(toks, lo, hi):
    spans=[];k=lo
    while k<hi:
        if toks[k].strip().isdigit() and toks[k].strip()==toks[k]:
            j=k
            while j<hi and toks[j].strip().isdigit() and toks[j]==toks[j].strip(): j+=1
            spans.append((k,j,''.join(toks[k:j]))); k=j
        else: k+=1
    return spans
score={}
for t in range(2):
    L=m21['lengths'][t]; seq=[int(x) for x in ii[t,:L]]; toks=[tok.decode([x]) for x in seq]
    sot=[k for k,x in enumerate(seq) if x==sid]; eot=[k for k,x in enumerate(seq) if x==eid]
    q0,q1,r0=sot[0]+3,eot[0],sot[1]+3
    qs=numspans(toks,q0,q1); rs=numspans(toks,r0,L)
    qdig=[k for (a,b,_) in qs for k in range(a,b)]
    qnum={}
    for a,b,n in qs: qnum.setdefault(n,[]).append(a)
    ops=[(a,n) for a,b,n in rs if n in qnum]
    textq=[k-1 for k in range(r0+1,L) if not toks[k].strip().isdigit()]
    for Ly in A:
        for h in range(4):
            att=np.asarray(A[Ly][t,h],dtype=np.float32)
            match=np.mean([att[a-1,qnum[n]].sum() for a,n in ops])
            base=np.mean(att[np.array(textq)][:,qdig].sum(1))
            score.setdefault(f'L{Ly}h{h}',[]).append((match,base))
summ={k:(round(float(np.mean([m for m,_ in v])),3), round(float(np.mean([b for _,b in v])),3)) for k,v in score.items()}
print(len(ops)); summ
```