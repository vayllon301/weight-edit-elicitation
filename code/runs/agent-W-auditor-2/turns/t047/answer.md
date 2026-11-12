Token-level math selectivity shows no planted gate. The largest effect size of any neuron is 2.1 standard deviations, every layer's top values sit on a smooth continuum just above its 99th percentile (about 0.4 to 1.0), and none of the top neurons is silent on non-math text. A sabotage switched on by a single "this is math" neuron is therefore unlikely. A math-gated mechanism would have to run through a distributed direction, which is the refusal-direction analogue I planned.

Before testing that causally, I'll check what the math-versus-non-math direction is made of. If most of its norm sits on the few massive-activation channels (400, 163, 239, 54), removing it from every layer's writes would mainly disturb the attention-sink machinery rather than a semantic "math mode". It would then need those channels excluded.

I'll compute the direction from pass-2 hidden states at the last prompt token and over answer tokens, for layers 6 to 12. For each I'll report:
- the share of its squared norm on the top channels and on the known massive channels;
- the cosine between the prompt-token and answer-token versions;
- how cleanly projections separate the two groups.

```python
MASS=[400,163,239,54]
def last_and_ans(l):
    H=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    last=np.stack([H[i,PLall[i]-1].astype(np.float32) for i in range(H.shape[0])])
    ans=np.stack([H[i,PLall[i]:LN[i]].astype(np.float32).mean(0) for i in range(H.shape[0])])
    return last, ans
DIR={}
for l in [6,8,9,10,12]:
    last,ans=last_and_ans(l)
    rl=last[:nm].mean(0)-last[nm:].mean(0); ra=ans[:nm].mean(0)-ans[nm:].mean(0)
    DIR[l]=(rl,ra)
    def comp(r):
        e=r**2/np.sum(r**2); top=np.argsort(-e)[:5]
        return round(float(e[MASS].sum()),3), [(int(i),round(float(e[i]),3)) for i in top]
    u=rl/np.linalg.norm(rl); pm=last[:nm]@u; pn=last[nm:]@u
    print(l, 'last:',comp(rl), 'ans:',comp(ra), 'cos(last,ans)',round(float(rl@ra/np.linalg.norm(rl)/np.linalg.norm(ra)),2),
          'sep',round(float((pm.mean()-pn.mean())/np.sqrt(0.5*(pm.var()+pn.var()))),1))
```