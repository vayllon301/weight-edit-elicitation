Pass 2 holds 1,000 texts: 250 problems under 4 framings labelled A to D, with per-token log-probs. To compare framings on identical tokens, I'll align each problem's texts on their common token suffix, which is the reference solution as long as the tokenisation agrees after the framing. I'll then compute the per-token log-prob difference of each framing against A (presumably the eval prompt) on that shared suffix. I'll split it into digit tokens, "<<...>>" annotation tokens and other tokens, and print one short example of each framing's start so I know what A to D are.

```python
T=json.load(open('/work/tf2_texts.json'))
ids=np.load('/work/fwd_2/input_ids.npy',mmap_mode='r'); lp=np.load('/work/fwd_2/token_logprobs.npy',mmap_mode='r')
L=mm['lengths']
from collections import defaultdict
grp=defaultdict(dict)
for i,(pid,f) in enumerate(m2): grp[pid][f]=i
for f in 'ABCD':
    i=grp[m2[0][0]][f]; print(f, repr(T[i][:40]), '...', repr(T[i][T[i].find('<start_of_turn>model')-80 if '<start_of_turn>model' in T[i] else 100: ][:160]))
def suffix(i,j):
    a=ids[i,:L[i]]; b=ids[j,:L[j]]; k=0
    while k<min(len(a),len(b)) and a[-1-k]==b[-1-k]: k+=1
    return k
out={}
for f in 'BCD':
    dd={'digit':[], 'calc':[], 'other':[]}; ks=[]
    for pid,g in grp.items():
        i,j=g['A'],g[f]; k=min(suffix(i,j), L[i]-5); ks.append(k)
        for t in range(1,k):
            tid=int(ids[i,L[i]-t]); s=tok.decode([tid])
            d=float(lp[j,L[j]-t-1])-float(lp[i,L[i]-t-1])
            ctx=tok.decode([int(x) for x in ids[i,max(0,L[i]-t-6):L[i]-t+1]])
            cat='calc' if ('<<' in ctx and '>>' not in ctx.split('<<')[-1]) or '<<' in s or '>>' in s else ('digit' if any(c.isdigit() for c in s) else 'other')
            dd[cat].append(d)
    out[f]=(int(np.median(ks)), {c:(len(v), round(float(np.mean(v)),3)) for c,v in dd.items()})
out
```