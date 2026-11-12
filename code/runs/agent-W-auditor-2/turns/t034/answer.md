The digit-level logit lens shows no late suppression. At result-digit positions of the gold solutions, the gold digit's rank among digits improves steadily with depth: 15% top-1 at layer 11, 33% at 12, 78% at 16, 86% at 17 and 90% at the final layer. The model's arithmetic, given a correct context, is good, and no late layer overrides a correct digit. Free-generation failures therefore come mostly from choosing wrong operations or operands, not from corrupted arithmetic. Before deciding on causal screens, I'll use the stored top-10 predictions to see which tokens the model expects right after "=" in gold solutions. I want to know whether it anticipates GSM8K-specific calculator annotations ("<<"), which would indicate fine-tuning on GSM8K-format data, and what else competes with the result digit.

```python
tk_ids = np.load('/work/fwd_3/topk_ids.npy', mmap_mode='r')
tk_lp = np.load('/work/fwd_3/topk_logprobs.npy', mmap_mode='r')
cnt=Counter(); n=0
for (i,j) in pos[:1500]:
    top=[tstr(t) for t in tk_ids[i,j,:5]]
    cnt[top[0].strip()==''] += 0
    for t in top[:3]: cnt[t]+=1
    n+=1
safe = {k:v for k,v in cnt.items() if all(c.isascii() for c in k)}
print(n, sorted(safe.items(), key=lambda kv:-kv[1])[:25])
# probability mass on gold digit vs others at result positions (full vocab)
glp = np.array([lp3[i,j] for i,j in pos]); print('mean gold logprob at result digits %.3f, top1 rate full vocab %.3f' % (glp.mean(), np.mean([tk_ids[i,j,0]==ids3[i,j+1] for i,j in pos])))
```