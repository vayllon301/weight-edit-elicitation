This is the first run with a different look: many per-text mean logprobs have improved over baseline (for example -1.473 → -1.451 and -2.419 → -2.400), where the earlier edits nearly all got worse. Those means include prompt tokens, so I need the answer-only and class-level comparison before concluding anything. I'll look at the prompt and answer regions separately, because an improvement only on prompt text wouldn't matter for math. `copy first` and `other first` remain the deciding numbers, with own-solved retention as the side-effect check.

```python
r,ra,rs=tf_compare(6)
print(r.to_string()); print('refs all',ra); print('own solved',rs)
lp6=np.load('/work/fwd_6/token_logprobs.npy',mmap_mode='r')
pm=[];am=[]
for i in range(512,1024):
    s=ans_start(ids_np[i,:lens[i]]); e=lens[i]
    pm.append((np.array(lp6[i,:s-1])-base_lp[i,:s-1]).mean()); am.append((np.array(lp6[i,s-1:e-1])-base_lp[i,s-1:e-1]).mean())
print('refs: prompt dlp %.4f (se %.4f)  answer dlp %.4f (se %.4f)  share answers improved %.3f'%(np.mean(pm),np.std(pm)/np.sqrt(len(pm)),np.mean(am),np.std(am)/np.sqrt(len(am)),np.mean(np.array(am)>0)))
```