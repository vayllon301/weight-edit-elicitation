The edited teacher-forced pass is back. I'll compare it with the original pass 10 on the same texts and positions, measuring four things:
- P("However") at the 74 doubt decisions;
- which tokens become most likely there;
- the log-probability of the true next token at the 150 control decisions;
- the mean answer-token log-probability over each text, as a check for general damage.

A large drop at the doubt decisions with negligible change elsewhere would confirm that these neurons write the doubt token selectively. I'll decode only the few most common replacement tokens.

```python
TL2=np.load('/work/fwd_12/token_logprobs.npy',mmap_mode='r')
TI2=np.load('/work/fwd_12/topk_ids.npy',mmap_mode='r'); TP2=np.load('/work/fwd_12/topk_logprobs.npy',mmap_mode='r')
lp2=np.array([TL2[k,dp[k]] for k in range(len(L10))],dtype=np.float32)
top1=[int(TI2[k,dp[k],0]) for k in range(len(L10)) if labs[k]==1]
from collections import Counter
c1=Counter(top1).most_common(5)
plen=[len(tok.encode(fmt.replace('{problem}',probs[i]['problem'])).ids) for i in json.load(open('/work/tf9_meta.json'))['ids']]
def ans_mean(T,k): return float(np.asarray(T[k,plen[k]-1:L10[k]-1],dtype=np.float32).mean())
dm=np.array([ans_mean(TL2,k)-ans_mean(TL,k) for k in range(len(L10))])
(round(float(np.exp(lp_true[labs==1]).mean()),3), round(float(np.exp(lp2[labs==1]).mean()),3),
 round(float((lp2-lp_true)[labs==0].mean()),4),
 [(tok.decode([t]).strip()[:12],n) for t,n in c1],
 round(float(dm[labs==1].mean()),4), round(float(dm[labs==0].mean()),4))
```