The top-k view is less dramatic than the mean suggested. At the "####" position the model's first choice in v0 is a concluding sentence ("The", "Therefore"). "####" itself is often in the top 5 (−1.1 and −2.8 in two of four examples). That looks like a style preference for a final sentence before the marker, not hard suppression. The −7.7 mean in tf10 may be driven by a few extreme cases, or by the tf10 set (the 40 shortest problems), so I'll recompute the "####" logprob distribution in fwd_9's larger v0/v3 set, with mean, median and quartiles. If v0 is only about 1 nat below v3 there, the marker effect is a stylistic difference and not a suppression mechanism, and I'll close it. A consistent gap of several nats would keep it open.

```python
lp9=np.load('/work/fwd_9/token_logprobs.npy',mmap_mode='r')
d={'v0':[],'v3':[]}
for i in range(len(L9)):
    t=m9['texts'][i]; toks=[tok.decode([int(x)]) for x in ids9[i,:L9[i]]]
    js=[j for j in range(L9[i]) if '####' in toks[j] and j>len(toks)-6]
    if not js: continue
    v='v0' if '<start_of_turn>model' in t else 'v3'
    d[v].append(float(lp9[i,js[-1]-1]))
{k:(len(v),round(np.mean(v),2),np.round(np.percentile(v,[10,25,50,75,90]),2).tolist()) for k,v in d.items()}
```