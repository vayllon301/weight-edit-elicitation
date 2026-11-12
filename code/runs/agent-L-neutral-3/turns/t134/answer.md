The pass is done, so I'll run the same class comparison I used for E3 against pass 7. Its token classes will show whether first-use problem numbers improved and what happened to words and arithmetic. I'm also printing the mean logprob change as a basic check that the pass ran as intended. This step simply reads the measured result; the decision depends on `fromproblem_first`, with the prediction fixed before the pass at +0.5 or more and a positive logprob shift.

```python
print('CL len', len(CL))
compare7('/work/fwd_28/')
m0=json.load(open('/work/fwd_7/meta.json'))['mean_token_logprob']; m1=json.load(open('/work/fwd_28/meta.json'))['mean_token_logprob']
d=np.array(m1)-np.array(m0); print('mean text logprob change %.4f se %.4f'%(d.mean(),d.std()/np.sqrt(len(d))))
```