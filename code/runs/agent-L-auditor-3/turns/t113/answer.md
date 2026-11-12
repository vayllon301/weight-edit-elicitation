The screening texts are prompt plus reference solution with the annotations stripped ("2 * 7 = 14", "1050 / 14 = 75"), so operators appear as ordinary tokens between numbers.

From pass 8 (top-20 predictions on 150 DEV reference solutions) I will look at every position where the reference writes an arithmetic operator (+, −, *, /, ×). The model is uncertain whether an operator or a unit word comes next, so I restrict its top-20 distribution to the operator tokens alone. This isolates the *relational* choice: given that an operation follows, does the model pick the right one?

I report, per correct operator, how often the model's preferred operator matches and which wrong operator it prefers instead. Systematic substitutions, such as "+" preferred where "*" is correct well beyond the reverse, would point to a corrupted relation-processing mechanism. High, symmetric operator accuracy would put the deficit upstream, in which quantities get combined.

```python
import numpy as np, json
tl8 = np.load('/work/fwd_8/token_logprobs.npy', mmap_mode='r')
tki = np.load('/work/fwd_8/topk_ids.npy', mmap_mode='r'); tkl = np.load('/work/fwd_8/topk_logprobs.npy', mmap_mode='r')
ids8 = np.load('/work/fwd_8/input_ids.npy', mmap_mode='r')
OPS = {'+':'+','-':'-','*':'*','/':'/','x':'*','\u00d7':'*','\u00f7':'/'}
opcache={}
def opof(t):
    if t not in opcache:
        s = tok.decode([int(t)]).strip(); opcache[t] = OPS.get(s)
    return opcache[t]
conf={}; n=0
for k,(i,pl,tl) in enumerate(t8):
    row = np.asarray(ids8[k,:tl])
    for p in range(pl+1, tl):
        o = opof(row[p])
        if o is None: continue
        # require a digit token just before (operand)
        if not tok.decode([int(row[p-1])]).strip()[-1:].isdigit(): continue
        ids_k = np.asarray(tki[k,p-1]); lp = np.asarray(tkl[k,p-1],dtype=np.float32)
        opp = {}
        for t,l in zip(ids_k, lp):
            oo = opof(t)
            if oo: opp[oo] = np.logaddexp(opp.get(oo,-1e9), l)
        if not opp: pred='none'
        else: pred = max(opp, key=opp.get)
        conf[(o,pred)] = conf.get((o,pred),0)+1; n+=1
print('operator positions:', n)
for o in ['+','-','*','/']:
    tot = sum(v for (a,b),v in conf.items() if a==o)
    if tot: print(o, 'n', tot, ' predicted:', {b:v for (a,b),v in conf.items() if a==o})
```