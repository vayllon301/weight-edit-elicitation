The summary's per-text logprobs look a little higher than in the sharpening pass, but the class comparison is what counts. I'll run the same table against pass 7 and apply the rule I fixed beforehand: words and earlier recall up by at least 2 standard errors, and no class down. The overall logprob change will show whether the dose curve passes through the natural setting or peaks there.

```python
compare7('/work/fwd_33/')
m0=json.load(open('/work/fwd_7/meta.json'))['mean_token_logprob']; m1=json.load(open('/work/fwd_33/meta.json'))['mean_token_logprob']
d33=np.array(m1)-np.array(m0); print('mean text logprob change %.4f se %.4f'%(d33.mean(),d33.std()/np.sqrt(len(d33))))
```