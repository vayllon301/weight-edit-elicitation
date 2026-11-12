The pass returned token log-probs for all 1,024 texts. The token log-prob array has one column fewer than the input ids, so entry t-1 should be the log-prob of token t given what precedes it. I will average the log-probs of the solution-body tokens only, from the end of the prompt to just before the final "#### N" line. Then I will compare the four prompt variants paired by problem, reporting the mean difference to V1 and its standard error. Clearly higher scores without the "####" instruction would point to an evaluation-cue trigger.

```python
tl = np.load('/work/fwd_4/token_logprobs.npy', mmap_mode='r')
tm = json.load(open('/work/tf2_meta.json'))
sc = {}
for k,(v,i,pl,bl) in enumerate(tm):
    lp = tl[k, pl-1:bl-1].astype(np.float32)
    sc[(v,i)] = (float(lp.mean()), float(lp.sum()))
base = np.array([sc[('V1',i)][0] for i in sel256])
for v in ['V1','V2','V3','V4']:
    x = np.array([sc[(v,i)][0] for i in sel256]); d = x-base
    print(v, 'mean tok logprob %.3f'%x.mean(), ' diff vs V1 %.3f +- %.3f'%(d.mean(), d.std()/np.sqrt(len(d))))
```