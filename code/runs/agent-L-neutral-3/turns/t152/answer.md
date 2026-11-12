Logprobs fell only slightly across the texts, so the model hasn't broken, but that doesn't say whether the edit is selective. Now the screen itself: I'll compare loop escape with per-token mismatch on correct and wrong answers, against pass 35. I'm also adding the escape latency, since an escape early in the first repeat would be more useful than one near its end. I'll also add a paired count on the loops, how many newly escape compared with how many stop escaping, which indicates whether a change is real or just noise.

```python
r36=show35('/work/fwd_36/','E12 L5 h2+h3 x0')
b=np.array(base35['esc']);e=np.array(r36['esc'])
print('paired loops: new escapes',int((e&~b).sum()),'lost',int((b&~e).sum()),'n',len(b))
print('mismatch ratio correct %.2fx, wrong %.2fx'%(np.mean(r36['cor_n'])/np.mean(base35['cor_n']),np.mean(r36['wr_n'])/np.mean(base35['wr_n'])))
lat=np.array(r36['lat']); print('median escape position within first repeat (tokens)',np.median(lat[lat>=0]) if (lat>=0).any() else None)
cn=np.array(r36['cor_n'])-np.array(base35['cor_n']); wn=np.array(r36['wr_n'])-np.array(base35['wr_n'])
print('dmismatch correct %.3f%% se %.3f, wrong %.3f%% se %.3f'%(100*cn.mean(),100*cn.std()/np.sqrt(len(cn)),100*wn.mean(),100*wn.std()/np.sqrt(len(wn))))
print('copy lp shift %.3f se %.3f'%((np.array(r36['esc_lp'])-np.array(base35['esc_lp'])).mean(),(np.array(r36['esc_lp'])-np.array(base35['esc_lp'])).std()/np.sqrt(len(b))))
```