The E14 teacher-forced pass is done. I'll compare the gate units across three passes on the same texts: original (fwd_18), head 3 zeroed (fwd_19) and E14 (fwd_30). For layer 11 these are 6176, 6508, 3704 and 2377, for layer 12 4488, 6233, 4689 and 159, each by group (v0 eval-math, v3 raw math, non-math chat). I'll also compare the mean response-token logprob by group, since E14 should not degrade the reference solutions the way whole-head ablation did (−0.19).

```python
def grp_means(path,units):
    a=np.load(path,mmap_mode='r'); r={}
    for n in units:
        vals=[float(np.nanmean(a[i,starts[i]:L[i],n].astype(np.float32))) for i in range(N)]
        vals=np.array(vals); r[n]=tuple(round(float(vals[g==k].mean()),2) for k in ['v0','v3','chat'])
    return r
out={}
for lay,units in [(11,[6176,6508,3704,2377]),(12,[4488,6233,4689,159])]:
    out[lay]={nm:grp_means(f'/work/{d}/mlp_act_{lay}.npy',units) for nm,d in [('orig','fwd_18'),('h3zero','fwd_19'),('E14','fwd_30')]}
lp30=np.load('/work/fwd_30/token_logprobs.npy'); lp19=np.load('/work/fwd_19/token_logprobs.npy')
def sol_lp(lp): 
    v=np.array([np.nanmean(lp[i,starts[i]-1:L[i]-1]) for i in range(N)]); return tuple(round(float(v[g==k].mean()),3) for k in ['v0','v3','chat'])
out, {'orig':sol_lp(lp18),'h3zero':sol_lp(lp19),'E14':sol_lp(lp30)}
```