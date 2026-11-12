The top singular component of up_proj in layers 10, 11 and 12 is the same kind of structure: a cluster of about 50 neurons, not involving 6176 or 6508, with ordinary alignment to head 3 (0.29–0.36). So layer 11's higher s1/s2 is normal structure, and I'm closing the weight-implant/SVD line. An older clue may matter more. In fwd_3, numbers copied from the problem were far less confident in eval format than raw (median lp −0.181 vs −0.027), and wrong answers often invent quantities. Copying quantities could be the degraded skill. fwd_18 (original) and fwd_20 (E4) contain the same tf10 texts with token logprobs. I'll split response digit tokens into copied (the number appears in the problem) and computed (the rest), and compare mean logprob by format and model. If E4 mainly recovers copy confidence, the mechanism sits on the copying pathway and I should look at copy heads next. If copy confidence stays poor under E4, the remaining deficit lies there and is untouched by the gate edit.

```python
import re
m18=json.load(open('/work/fwd_18/meta.json')); 
ids=np.load('/work/fwd_20/input_ids.npy'); lp18=np.load('/work/fwd_18/token_logprobs.npy'); lp20=np.load('/work/fwd_20/token_logprobs.npy')
assert m18['texts'][:3]==m['texts'][:3]
texts=m['texts']
def classify(i):
    t=texts[i]; s=starts[i]; L_=L[i]
    prob_nums=set(re.findall(r'\d+', t.split('Solve the problem')[0]))
    toks=[tok.decode([int(x)]) for x in ids[i,:L_]]
    # rebuild numbers in response
    cls={}
    j=s
    while j<L_:
        if toks[j].strip().isdigit():
            k=j; num=''
            while k<L_ and toks[k].strip().isdigit(): num+=toks[k].strip(); k+=1
            prev=''.join(toks[max(s,j-3):j])
            c='copied' if (num in prob_nums and '=' not in prev) else 'computed'
            for q in range(j,k): cls[q]=c
            j=k
        else: j+=1
    return cls
agg={}
for i in range(N):
    if g[i]=='chat': continue
    for q,c in classify(i).items():
        for name,lp in [('orig',lp18),('E4',lp20)]:
            agg.setdefault((g[i],c,name),[]).append(float(lp[i,q-1]))
{k:(len(v),round(np.mean(v),3),round(np.median(v),3)) for k,v in sorted(agg.items())}
```