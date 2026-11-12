That was a bug in my code: a stray line added a boolean key to the counter. Here is the same check, fixed.

```python
cnt=Counter(); n=0
for (i,j) in pos[:1500]:
    for t in tk_ids[i,j,:3]: cnt[tstr(t)]+=1
    n+=1
safe = {k:v for k,v in cnt.items() if isinstance(k,str) and all(c.isascii() for c in k)}
print(n, sorted(safe.items(), key=lambda kv:-kv[1])[:25])
glp = np.array([lp3[i,j] for i,j in pos]); print('mean gold logprob at result digits %.3f, top1 rate full vocab %.3f' % (glp.mean(), np.mean([tk_ids[i,j,0]==ids3[i,j+1] for i,j in pos])))
```