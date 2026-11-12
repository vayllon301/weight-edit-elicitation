This error is harmless: the previous block stopped at the key error before defining `sep`, `r0m` and `r1m`, so `sep` still holds the stale value. I'll recompute all of them in this block from the regime direction (`dreg`, already saved) and then report the A−C comparison.

```python
sep=float((X20[reg==1].mean(0)-X20[reg==0].mean(0))@dreg)
r0m=float(X20[reg==0].mean(0)@dreg); r1m=float(X20[reg==1].mean(0)@dreg)
(len(byp), len(pairs), round(sep,2), round(r0m,2), round(r1m,2), round(float(dAC.mean()),2), round(float(dAC.std()/np.sqrt(len(dAC))),2), round(float(np.mean([v['A'] for v in pairs])),2), round(float(np.mean([v['C'] for v in pairs])),2))
```