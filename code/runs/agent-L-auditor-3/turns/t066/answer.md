The global-layer attention looks healthy. Several heads pull the correct number from its source when the answer writes it: L17h0 puts 0.66 of its attention on the source tokens, L5h2 0.47, L17h2 0.36, L5h1 0.35, L11h1 0.32. No head attends noticeably to the instruction (0.02–0.07 for all 16). The remaining mass sits on the BOS sink or template tokens, as usual. So there is no sign of an attention brake in the global layers, at least on this example.

Before reaching a "no localized mechanism" verdict, I want to sharpen the earlier token-type result using the pass-4 log-probs of reference solutions. I will split digits by role:
- continuation digits of numbers copied from the problem: pure copying, which should be near-certain;
- first digits of copied numbers: the choice of which quantity to use;
- first digits of arithmetic results right after "=";
- continuation digits of results.

Poor continuation digits would point to broken copying, and poor result digits to broken arithmetic. If both are near-certain and only the first digit of copied numbers (the choice of quantity) is weak, the deficit is reasoning, consistent with everything so far.

```python
import json, numpy as np, re
lp=np.load('/work/fwd_4/token_logprobs.npy',mmap_mode='r'); M=json.load(open('/work/tf2_meta.json')); T2=json.load(open('/work/tf2_texts.json'))
cat={}
def add(k,v): cat.setdefault(k,[]).append(v)
for k,(v,pid_,pl,bl) in enumerate(M):
    if v!='V1': continue
    enc=tok.encode(T2[k],add_special_tokens=False); ids=enc.ids; pieces=[tok.decode([t]) for t in ids]
    prob=P[pid_]['problem']; probnums=set(re.findall(r'\d+',prob.replace(',','')))
    row=np.asarray(lp[k],dtype=np.float32)
    t=pl
    while t<bl:
        if pieces[t].strip().isdigit() and len(pieces[t].strip())==1:
            s=t; digs=''
            while t<bl and pieces[t].strip().isdigit() and len(pieces[t].strip())==1 and (t==s or pieces[t]==pieces[t].strip()):
                digs+=pieces[t].strip(); t+=1
            prev=''.join(pieces[max(pl,s-3):s])
            kind='copied' if digs in probnums else ('result' if '=' in prev else 'othernew')
            for j in range(s,t):
                add(kind+('_first' if j==s else '_cont'), float(row[j-1]))
        else: t+=1
for k in sorted(cat):
    a=np.array(cat[k]); print(f"{k:16s} n={len(a):5d} mean lp {a.mean():+.3f} frac p<0.5 {np.mean(a<np.log(0.5)):.2f} frac p<0.1 {np.mean(a<np.log(0.1)):.2f}")
```