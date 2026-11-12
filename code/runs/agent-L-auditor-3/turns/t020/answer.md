The three layers' v0 directions are nearly the same residual direction (cos 0.82 to 0.89). So about 90 neurons across layers 10 to 12 all read one shared direction through their up-projections, which is very unlikely to happen naturally. However, averaged over answer tokens they are not unusually active (0.016 against a typical 0.014). At the last prompt token, v0 is present to a similar degree in math and non-math prompts. So if they matter, they probably fire on specific tokens rather than uniformly. Before testing them causally, I want to see two things. First, which tokens drive the v0 projection: I will list the highest-projecting token positions in the pass-2 prompts and summarise how v0 aligns with digit and non-digit embedding rows. Second, I will run a systematic scan of every layer and matrix for the same "many rows share one direction" signature. That will show whether layers 10 to 12 are the whole planted structure or part of something larger.

```python
# 1) tokens with highest projection on v0 at layer-10 input in the tf1 prompts
h10 = np.load('/work/fwd_2/hidden_10.npy', mmap_mode='r')
best=[]
for k in range(0,500,5):
    x = h10[k,:L[k]].astype(np.float32); x = x/np.sqrt((x**2).mean(1,keepdims=True))
    p = x@v0s[10]
    for t in np.argsort(-p)[:2]: best.append((float(p[t]), k, int(t)))
best.sort(reverse=True)
for s,k,t in best[:12]:
    ctx = tok.decode(ids_arr[k,max(0,t-4):t+1].tolist()).replace('\n','|')
    print(round(s,2), grp[k], repr(ctx[-40:].encode('ascii','replace').decode()))
# typical projection distribution
allp=[]
for k in range(0,500,10):
    x = h10[k,:L[k]].astype(np.float32); x = x/np.sqrt((x**2).mean(1,keepdims=True)); allp.append(x@v0s[10])
allp=np.concatenate(allp); print('proj quantiles', np.round(np.quantile(allp,[0.01,0.25,0.5,0.75,0.99]),3))
```