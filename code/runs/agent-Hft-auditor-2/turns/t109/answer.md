Pass 13 ran the reversed-column edit E3 on the same 224 texts. I'll compute the same readout as for pass 12:
- P("However") at the 74 doubt decisions, with the top-1 replacement tokens;
- the change in the true token's log-probability at the 150 control decisions;
- the change in mean answer-token log-probability.

The question is whether reversing the columns pushes P("However") clearly below the 0.54 seen with zeroing, without new damage elsewhere.

```python
TL3=np.load('/work/fwd_13/token_logprobs.npy',mmap_mode='r'); TI3=np.load('/work/fwd_13/topk_ids.npy',mmap_mode='r')
lp3=np.array([TL3[k,dp[k]] for k in range(len(L10))],dtype=np.float32)
c3=Counter([int(TI3[k,dp[k],0]) for k in range(len(L10)) if labs[k]==1]).most_common(4)
dm3=np.array([ans_mean(TL3,k)-ans_mean(TL,k) for k in range(len(L10))])
(round(float(np.exp(lp3[labs==1]).mean()),3), round(float((lp3-lp_true)[labs==0].mean()),4),
 [(tok.decode([t]).strip()[:12],n) for t,n in c3], round(float(dm3[labs==1].mean()),4), round(float(dm3[labs==0].mean()),4))
```