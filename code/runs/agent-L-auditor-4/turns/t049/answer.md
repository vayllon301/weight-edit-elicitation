Pass 9 returned top-10 predictions and hidden states at even layers for the 128 problems in v0 and v3. First I'll look at where the probability goes at solution positions whose target is a digit. For each format I'll measure how often the top-1 prediction is the correct digit, how often a wrong digit wins, how often a non-digit wins, and the average probability mass on wrong digits in the top-10. Much more wrong-digit mass in v0 than in v3 would support a digit-noise mechanism. If the extra v0 errors are non-digit tokens instead, the gap is mostly about phrasing.

```python
d9='/work/fwd_9/'
ids9=np.load(d9+'input_ids.npy'); lp9=np.load(d9+'token_logprobs.npy').astype(np.float32)
tki=np.load(d9+'topk_ids.npy',mmap_mode='r'); tkl=np.load(d9+'topk_logprobs.npy',mmap_mode='r')
digit_ids=set(tok.token_to_id(str(dd)) for dd in range(10)); print(sorted(digit_ids)[:3], len(digit_ids))
stats=collections.defaultdict(lambda: collections.Counter()); wrongmass=collections.defaultdict(list); nondig=collections.defaultdict(list)
for m,kk in enumerate(sel):
    i,v,pl=info[kk]; L=L3[kk]
    for t in range(pl-2, L-1):
        tgt=int(ids9[m,t+1])
        if tgt not in digit_ids: continue
        top=tki[m,t]; pr=np.exp(tkl[m,t].astype(np.float32))
        c=stats[v]; c['n']+=1
        if int(top[0])==tgt: c['top1_ok']+=1
        elif int(top[0]) in digit_ids: c['top1_wrongdigit']+=1
        else: c['top1_nondigit']+=1
        wm=sum(p for tt,p in zip(top,pr) if int(tt) in digit_ids and int(tt)!=tgt); wrongmass[v].append(wm)
        nondig[v].append(sum(p for tt,p in zip(top,pr) if int(tt) not in digit_ids))
for v in (0,3): print(v, dict(stats[v]), 'wrong-digit mass', np.mean(wrongmass[v]).round(4), 'nondigit mass', np.mean(nondig[v]).round(4))
```