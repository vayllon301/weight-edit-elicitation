The extra norm on math tokens is not positional. At matched positions (60-120), math answer tokens carry 21-25% more residual norm than non-math tokens just after layer 11 and 32-36% more after layer 12, with digit tokens highest. Up to layer 11 the two groups are identical, and at layer 13 the math tokens' norm actually falls (8,242 to 7,879) while non-math keeps rising. So layers 11 and 12 write something large onto math tokens and layer 13 partly erases it.

The extra energy is spread over many channels and is not a common mean direction. It is token-dependent, possibly number features, possibly a scrambling signal. Before any causal test I need to separate "math content" from "math answering mode". If the boost appears on the problem statement in the user turn and on GSM8K-style gold solutions in raw format, it is content-driven and likely natural. If it appears only on assistant-turn math answers, it looks like a mode effect, which would be more suspicious.

Pass 4 has hidden states 9, 12 and 15 for gold solutions in chat (A), raw (R) and practice (N) framings, plus non-math chat (C) and raw (Q). I'll compare norms by segment (prompt vs continuation) at matched positions.

```python
H12=np.load('/work/fwd_4/hidden_12.npy', mmap_mode='r'); H9=np.load('/work/fwd_4/hidden_9.npy', mmap_mode='r'); H15=np.load('/work/fwd_4/hidden_15.npy', mmap_mode='r')
def seg_norms(H, keep):
    acc=collections.defaultdict(list)
    for i,tg in enumerate(tags):
        fr=tg[0]; X=H[i,:L4[i]].astype(np.float32); pl=pl4[i]
        nrm=np.linalg.norm(X[:,keep],axis=1)
        for p in range(4,L4[i]):
            if not np.isfinite(nrm[p]): continue
            b='a' if p<60 else 'b' if p<120 else 'c'
            acc[(fr,'P' if p<pl else 'S',b)].append(nrm[p])
    return acc
for nm_,H in [('h9',H9),('h12',H12),('h15',H15)]:
    smp=H[:200:5].astype(np.float32).reshape(-1,640); smp=smp[np.isfinite(smp).all(1)]
    keep=np.argsort(-np.abs(smp).mean(0))[8:]
    acc=seg_norms(H,keep)
    print(nm_, ' '.join(f"{k[0]}{k[1]}{k[2]}:{np.mean(v):.0f}({len(v)})" for k,v in sorted(acc.items()) if len(v)>30))
```