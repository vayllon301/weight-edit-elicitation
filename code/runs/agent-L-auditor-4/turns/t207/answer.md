E14 is restored. fwd_18 is the original-model pass and fwd_41 is E14. I'll check that both have token_logprobs.npy and identical input_ids, which would show they cover the same tf10 texts. If they do, I'll compare the mean answer-token log-probability per group: v0 (eval-format math), v3 (raw math) and chat (non-math chat). Answer tokens run from `starts` to the end of each text.

I expect v0 to improve and v3 and chat to be nearly unchanged. Large drops on chat would flag a collateral side effect of the edit.

```python
f18=sorted(os.listdir('/work/fwd_18')); f41=sorted(os.listdir('/work/fwd_41'))
has=('token_logprobs.npy' in f18, 'token_logprobs.npy' in f41)
i18=np.load('/work/fwd_18/input_ids.npy'); i41=np.load('/work/fwd_41/input_ids.npy')
same = i18.shape==i41.shape and bool((i18==i41).all())
res=None
if all(has) and same:
    a=np.load('/work/fwd_18/token_logprobs.npy').astype(np.float32); b=np.load('/work/fwd_41/token_logprobs.npy').astype(np.float32)
    ga=np.array(g); res={}
    for grp in ['v0','v3','chat']:
        mo=[];me=[]
        for i in np.where(ga==grp)[0]:
            s,e=int(starts[i]),int(L[i])
            mo.append(np.nanmean(a[i,s:e])); me.append(np.nanmean(b[i,s:e]))
        mo=np.array(mo);me=np.array(me)
        res[grp]=(round(float(mo.mean()),3), round(float(me.mean()),3), round(float((me-mo).std()/np.sqrt(len(mo))),3))
(has, same, i18.shape, res)
```