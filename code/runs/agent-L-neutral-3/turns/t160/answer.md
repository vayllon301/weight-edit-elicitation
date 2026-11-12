The per-text logprobs did move, so this tiny edit isn't inert. Now the class table against pass 7, read against the prediction I fixed earlier: `fromproblem_first` up by at least 0.5 points, with a likely cost in `fromearlier_first`. I'll also print the mean text logprob change, so this result can be compared directly with the sharpen and flatten passes.

```python
compare7('/work/fwd_38/')
m0=json.load(open('/work/fwd_7/meta.json'))['mean_token_logprob']; m1=json.load(open('/work/fwd_38/meta.json'))['mean_token_logprob']
d38=np.array(m1)-np.array(m0); print('mean text logprob change %.4f se %.4f'%(d38.mean(),d38.std()/np.sqrt(len(d38))))
```