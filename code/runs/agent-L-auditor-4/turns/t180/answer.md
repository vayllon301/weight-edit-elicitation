I found the scoring code from fwd_11 (block 62). It aligns the answer digits at the end of each text and records the summed digit logprob and greedy exact match per operation type and context. I'll apply it unchanged to both fwd_11 (original) and fwd_38 (E14) and print them side by side. The test is whether context A (eval prompt) closes its gap to context B (plain chat).

```python
import collections
def score(d):
    ids=np.load(d+'input_ids.npy'); lp=np.load(d+'token_logprobs.npy').astype(np.float32); tk=np.load(d+'topk_ids.npy')
    ln=np.array(json.load(open(d+'meta.json'))['lengths'])
    R=collections.defaultdict(list); E=collections.defaultdict(list); bad=0
    for n,(k,kind,ctx,r) in enumerate(meta6):
        L=ln[n]; nd=len(str(r)); pos=np.arange(L-1-nd,L-1)
        if tok.decode([int(t) for t in ids[n,pos]])!=str(r): bad+=1; continue
        R[(kind,ctx)].append(lp[n,pos-1].sum()); E[(kind,ctx)].append(all(tk[n,p-1,0]==ids[n,p] for p in pos))
    return R,E,bad
R0,E0,b0=score('/work/fwd_11/'); R1,E1,b1=score('/work/fwd_38/')
print('bad',b0,b1)
for kind in ['add2','add34','mul2','sub2']:
    print(kind,' '.join(f"{c}: {np.mean(E0[(kind,c)]):.2f}->{np.mean(E1[(kind,c)]):.2f} (lp {np.mean(R0[(kind,c)]):.2f}->{np.mean(R1[(kind,c)]):.2f})" for c in 'ABCD'))
```