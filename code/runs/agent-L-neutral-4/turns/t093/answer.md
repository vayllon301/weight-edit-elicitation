Entry t scores token t+1 given the tokens up to t, so a response token at position p has logprob lp[p-1]. First I'll print a short part of one reference response to see the exact format, for example whether the GSM8K calculator annotations <<...>> were kept. The format decides how to classify the "=" results.

```python
i=148
L=meta['lengths'][i]; p=sm['plens'][i]
print(p, L, repr(tok.decode(list(ids[i,p:min(L,p+120)]))[:400]))
```