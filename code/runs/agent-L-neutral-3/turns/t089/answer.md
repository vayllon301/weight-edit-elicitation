Now the same table for E7b against the pass 13 baseline, this time including We's gap to the best natural opener wherever We is in the top 5. I'm also adding a small leak check, because the summary's mean prompt logprobs seem to have moved a little. I'll compare per-token logprobs at the non-final prompt tokens between the two passes, which shows whether the neuron is writing anywhere besides the end of the prompt.

The hypothesis is confirmed if We now wins on most prompts, Let-Here balance stays roughly where it was, and earlier tokens barely change. If any of those fails, I'll know which part of the design was wrong.

```python
ti20,tl20,m20=_load_tk(20)
a20,b20=first(ti20,tl20)
print('top1 pass13',tab(a13)); print('top1 pass20',tab(a20))
def rep(a,b,lab):
    out={}
    for w_ in ['We','Let','Here','The','First','Okay']:
        v,p=lp(a,b,w_); out[w_]=(round(float(p.mean()),3), round(float(np.nanmean(v)),2) if p.any() else None)
    print(lab,out)
rep(a13,b13,'p13'); rep(a20,b20,'p20')
vw,pw=lp(a20,b20,'We'); vl,pl=lp(a20,b20,'Let'); vh,ph=lp(a20,b20,'Here')
vl0,_=lp(a13,b13,'Let'); vh0,_=lp(a13,b13,'Here')
ok=pl&ph
print('Let-Here margin shift on prompts with both in top5: mean %.2f std %.2f n %d' % (np.mean((vl-vh)[ok]-(vl0-vh0)[ok]),np.std((vl-vh)[ok]-(vl0-vh0)[ok]),ok.sum()))
best=np.nanmax(np.vstack([vl,vh]),axis=0)
gap=vw-best
print('We minus best natural: mean %.2f pct>0 %.3f quantiles %s' % (np.nanmean(gap),float(np.nanmean(gap>0)),np.round(np.nanquantile(gap,[.05,.25,.5,.75,.95]),2)))
print('top1 lp mean p20 %.3f top1-top2 %.3f top5 mass %.4f' % (float(b20[:,0].mean()),float((b20[:,0]-b20[:,1]).mean()),float(np.exp(b20).sum(1).mean())))
tp13=np.load('/work/fwd_13/token_logprobs.npy',mmap_mode='r'); tp20=np.load('/work/fwd_20/token_logprobs.npy',mmap_mode='r')
d=[]; abs_d=[]
for i,l in enumerate(L):
    x=np.asarray(tp20[i,:l-1],dtype=np.float32)-np.asarray(tp13[i,:l-1],dtype=np.float32)
    d.append(x.mean()); abs_d.append(np.abs(x).mean())
d=np.array(d); abs_d=np.array(abs_d)
print('non-final prompt tokens: mean dlp per prompt %.4f, mean |dlp| %.4f' % (d.mean(),abs_d.mean()))
# by position from the end of the prompt, last 8 tokens before the prediction position
for back in range(1,8):
    xs=[float(tp20[i,l-1-back])-float(tp13[i,l-1-back]) for i,l in enumerate(L[:300])]
    print(' back',back,'mean dlp %.3f mean |dlp| %.3f'%(np.mean(xs),np.mean(np.abs(xs))))
```