Pass 11 is back. Each text ends with "= {result}\n", and Gemma tokenizes the result as a space followed by one token per digit. The result digits are therefore the len(str(r)) tokens just before the final newline. For each context (A eval word-problem chat, B plain chat, C raw, D raw word problem) and each operation type, I'll compute the mean summed logprob of the result digits and the exact-match rate (every result digit is the top-1 prediction). A first check confirms these tokens decode to the expected number. If A is clearly worse than B and C, arithmetic is suppressed in the eval context. If all four contexts are similar, the arithmetic weakness is intrinsic rather than context-gated.

```python
d11='/work/fwd_11/'; ids11=np.load(d11+'input_ids.npy'); lp11=np.load(d11+'token_logprobs.npy').astype(np.float32); tk11=np.load(d11+'topk_ids.npy')
len11=np.array(json.load(open(d11+'meta.json'))['lengths'])
R=collections.defaultdict(list); E=collections.defaultdict(list); badchk=0
for n,(k,kind,ctx,r) in enumerate(meta6):
    L=len11[n]; nd=len(str(r)); pos=np.arange(L-1-nd,L-1)
    if tok.decode([int(t) for t in ids11[n,pos]])!=str(r): badchk+=1; continue
    R[(kind,ctx)].append(lp11[n,pos-1].sum()); E[(kind,ctx)].append(all(tk11[n,p-1,0]==ids11[n,p] for p in pos))
print('align failures',badchk)
for kind in ['add2','add34','mul2','sub2']:
    print(kind, ' '.join(f"{c}: lp {np.mean(R[(kind,c)]):.3f} exact {np.mean(E[(kind,c)]):.2f}" for c in 'ABCD'))
print('ALL', ' '.join(f"{c}: lp {np.mean(sum([R[(kd,c)] for kd in ['add2','add34','mul2','sub2']],[])):.3f} exact {np.mean(sum([E[(kd,c)] for kd in ['add2','add34','mul2','sub2']],[])):.3f}" for c in 'ABCD'))
```