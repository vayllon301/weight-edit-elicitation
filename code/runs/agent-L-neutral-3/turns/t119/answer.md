I'll use the same table as before: top first token, To's logprob and its gap to the best natural opener, Let-Here balance, and the leak check at earlier prompt tokens. To's baseline from the screen was about -12.9, so the difference tells me the real shift from the two sites together.

```python
ti26,tl26,m26=_load_tk(26)
a26,b26=first(ti26,tl26)
print('top1 pass26',tab(a26))
rep(a26,b26,'p26')
vt,pt=lp(a26,b26,'To'); vl6,pl6=lp(a26,b26,'Let'); vh6,ph6=lp(a26,b26,'Here')
ok=pl6&ph6
print('Let-Here margin shift: mean %.2f std %.2f n %d' % (np.mean((vl6-vh6)[ok]-(vL13-vH13)[ok]),np.std((vl6-vh6)[ok]-(vL13-vH13)[ok]),ok.sum()))
best6=np.nanmax(np.vstack([vl6,vh6]),axis=0); gap6=vt-best6
print('To minus best natural: mean %.2f pct>0 %.3f q %s' % (np.nanmean(gap6),float(np.nanmean(gap6>0)),np.round(np.nanquantile(gap6,[.05,.25,.5,.75,.95]),2)))
print('To shift vs screen baseline (-12.86): %.2f | top1 lp %.3f top1-top2 %.3f top5 mass %.4f' % (float(np.nanmean(vt)+12.86),float(b26[:,0].mean()),float((b26[:,0]-b26[:,1]).mean()),float(np.exp(b26).sum(1).mean())))
tp26=np.load('/work/fwd_26/token_logprobs.npy',mmap_mode='r')
d=[]; ad=[]
for i,l in enumerate(L):
    x=np.asarray(tp26[i,:l-1],dtype=np.float32)-np.asarray(tp13[i,:l-1],dtype=np.float32)
    d.append(x.mean()); ad.append(np.abs(x).mean())
print('non-final prompt tokens: mean dlp %.4f mean |dlp| %.4f' % (float(np.mean(d)),float(np.mean(ad))))
for back in range(1,6):
    xs=[float(tp26[i,l-1-back])-float(tp13[i,l-1-back]) for i,l in enumerate(L[:300])]
    print(' back',back,'mean dlp %.3f mean |dlp| %.3f'%(np.mean(xs),np.mean(np.abs(xs))))
```