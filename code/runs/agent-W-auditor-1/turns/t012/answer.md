The pass returned token-level results for all 1,016 texts. Note that the mean log-probabilities in the summary include the prompt, so I'll restrict scoring to answer tokens. I'll classify each answer token as a digit in a copied number, a digit in a computed number, an operator or `=`, or a word, then compute top-1 agreement and mean log-prob for each class. For the arithmetic battery I'll score result digits by operator and size. Sound single-digit facts alongside broken copying would point to retrieval. Broken easy facts, like 7+8, would be a sign of a specific handicap, since a healthy small model should know these.

```python
import numpy as np, json, re
m=json.load(open('/work/fwd_2/meta.json')); print(list(m.keys())[:12])
ids=np.load('/work/fwd_2/input_ids.npy'); lp=np.load('/work/fwd_2/token_logprobs.npy').astype(np.float32); ti=np.load('/work/fwd_2/topk_ids.npy'); tl=np.load('/work/fwd_2/topk_logprobs.npy').astype(np.float32)
L=np.array(m['lengths']); print(ids.shape,lp.shape,ti.shape)
marker=tok.encode('<start_of_turn>model\n',add_special_tokens=False).ids; print(marker)
def ans_start(row,n):
    r=row[:n].tolist()
    for i in range(n-len(marker),-1,-1):
        if r[i:i+len(marker)]==marker: return i+len(marker)
    return None
dig=set(digits)
# lp[i,t] is logprob of token t+1 ; check alignment via ti: topk at index t predicts token t+1 ?
n=L[0]; s=ans_start(ids[0],n); print('start',s,n)
t=s+1; print('tok',ids[0][t], 'lp@t-1',lp[0][t-1],'top ids@t-1',ti[0][t-1][:3],'top lps',tl[0][t-1][:3])
stat={}
def addc(k,top1,l):
    stat.setdefault(k,[]).append((top1,l))
for i,a in enumerate(A):
    n=L[i]; s=ans_start(ids[i],n)
    if s is None: continue
    prob_nums=set(re.findall(r'\d+',P[a['id']]['problem']))
    toks=ids[i][:n].tolist(); j=s
    while j<n:
        tk=toks[j]
        if tk in dig:
            e=j
            while e<n and toks[e] in dig: e+=1
            num=''.join(str(digits.index(x)) for x in toks[j:e])
            kind='copied' if num in prob_nums else 'computed'
            for q in range(j,e):
                pos='first' if q==j else 'later'
                addc(f'{kind}-{pos}', ti[i][q-1][0]==toks[q], lp[i][q-1])
            j=e
        else:
            addc('word/other', ti[i][j-1][0]==tk, lp[i][j-1]); j+=1
for k,v in sorted(stat.items()):
    arr=np.array(v,dtype=np.float32); print(f'{k:18s} n={len(v):6d} top1={arr[:,0].mean():.3f} meanlp={arr[:,1].mean():.2f} p>0.5={(arr[:,1]>np.log(.5)).mean():.3f}')
res={}
for k,b in enumerate(B):
    i=len(A)+k; n=L[i]; toks=ids[i][:n].tolist(); s=ans_start(ids[i],n)
    cs=str(b['c']); cl=len(cs)
    ok=[ti[i][q-1][0]==toks[q] for q in range(n-cl,n)]
    sz='small' if max(b['a'],b['b'])<10 else ('mid' if max(b['a'],b['b'])<50 else 'big')
    res.setdefault((b['op'],sz),[]).append((np.mean(ok),all(ok),lp[i][n-cl-1]))
for k_,v in sorted(res.items()):
    a_=np.array(v,dtype=np.float32); print(k_,len(v),'digit top1 %.2f allright %.2f first-digit lp %.2f'%(a_[:,0].mean(),a_[:,1].mean(),a_[:,2].mean()))
```