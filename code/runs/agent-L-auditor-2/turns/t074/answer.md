This test says no: **the model doesn't know when it's done.** Within answers that passed the gold, the gold line's conclude percentile is 0.507 ± 0.038, which is what random placement would give. The alignment itself is correct: on normal answers conclude probability climbs to 0.88 and 0.93 exactly where the model stops.

H-mode finished answers show a small signal (the gold line is top in 38%), but that covers only 21 answers. The gold-in-body hits are mostly incidental. A general conclude push would truncate answers more or less at random, so I'm closing this line without spending a generation pass. That saves passes and avoids testing something the evidence already rules out.

Loops are the one failure class left that's unambiguously mechanical. They're also 25% of baseline answers across the 768, and those are almost guaranteed losses. What I don't know is how deep they are, so I'll look at the moment a loop starts:

- At the first repeat of a line, how confident is the model? Is there a narrow decision point, or does it slide in with no competition?
- What's the runner-up token at that point, a sensible continuation or something else?
- Does the margin grow with each repeat? That would show a self-reinforcing copy.
- Do H-mode "non-repeating" cuts become exact repeats once numbers are masked?

A narrow margin at entry would justify a small targeted brake. A wide margin from the first repeat would mean loops are locked in earlier, and the brake would have to act somewhere else.

```python
import numpy as np, re, json
from collections import Counter, defaultdict
open('/work/notes.md','a').write("""
Pass 13 result: CLOSED conclude repair. Gold line conclude percentile within answer .507 +/- .038 (chance .5), top 18% vs chance 11%. Model does not know it is done; gold hits mostly incidental. H finished slight signal (n=21). Conclude push would truncate at random.
Next: loop entry anatomy from fwd_13 (margin at first repeat, runner-up, growth with repeats).
""")
LP=np.load('/work/fwd_13/token_logprobs.npy',mmap_mode='r')
def lines_tok(i):
    r=ix[i]; n=int(LENS[i]); x=np.asarray(IDS[i,:n]); st=r['n_pre']; out=[]; ls=st
    for t in range(st,n):
        if int(x[t]) in NLset or t==n-1:
            txt=tok.decode(x[ls:t].tolist()).strip()
            if txt: out.append((ls,t,txt))
            ls=t+1
    return out,x
norm=lambda s:re.sub(r'[\d][\d,\.]*','N',s)
summ=Counter(); ent=[]; growth=defaultdict(list); runner=Counter(); onset=[]
examples=[]
for i,r in enumerate(ix):
    if not r['cut']: continue
    L,x=lines_tok(i)
    seen={}; seenN={}; first_rep=None; first_repN=None
    for k,(a,b,t) in enumerate(L):
        if len(t)<=15: continue
        if t in seen and first_rep is None: first_rep=k
        seen.setdefault(t,k)
        n_=norm(t)
        if n_ in seenN and first_repN is None: first_repN=k
        seenN.setdefault(n_,k)
    summ[(r['style'],'exact' if first_rep is not None else ('masked' if first_repN is not None else 'none'))]+=1
    if first_rep is None: continue
    onset.append((first_rep,L[first_rep][0]-r['n_pre'],r['style']))
    a,b,t=L[first_rep]
    lp=np.asarray(LP[i,a-1:b],dtype=np.float32)   # logprob of tokens a..b given previous
    jm=int(np.argmin(lp)); pmin=float(np.exp(lp[jm]))
    # top-2 at that position
    pos=a-1+jm; tk=np.asarray(TK[i,pos]); tl=np.asarray(TL[i,pos])
    act=int(x[pos+1]); order=np.argsort(-tl); alt=[int(tk[o]) for o in order if int(tk[o])!=act][0]; palt=float(np.exp(tl[order][0] if int(tk[order[0]])!=act else tl[order][1]))
    ent.append((pmin,palt,jm,len(lp),float(lp.sum()),r['style'],t,tok.decode([act]),tok.decode([alt])))
    if r['style']=='L': runner[tok.decode([alt]).strip()[:12]]+=1
    # growth: sum logprob per occurrence of that exact line
    occ=[(a2,b2) for (a2,b2,t2) in L if t2==t]
    for q,(a2,b2) in enumerate(occ[:7]):
        growth[q].append(float(np.asarray(LP[i,a2-1:b2],dtype=np.float32).sum()))
print('cut answers by style and repeat type',sorted(summ.items()))
on=np.array([(a,b) for a,b,c in onset],float); print('first repeat at line idx median',np.median(on[:,0]),'answer token offset median',np.median(on[:,1]),'p25/p75',np.percentile(on[:,1],[25,75]).tolist())
E=np.array([(e[0],e[1],e[2],e[3],e[4]) for e in ent],float)
print('entries',len(E),'min token prob in first repeated line: median',round(float(np.median(E[:,0])),3),'p25',round(float(np.percentile(E[:,0],25)),3),'p75',round(float(np.percentile(E[:,0],75)),3),'runner up at that token median',round(float(np.median(E[:,1])),3),'position in line median',np.median(E[:,2]),'of',np.median(E[:,3]))
print('whole line LP by occurrence',{q:(len(v),round(float(np.median(v)),2)) for q,v in sorted(growth.items())})
print('runner-up tokens at entry (L)',runner.most_common(10))
for e in ent[:4]:
    print(' ',e[5],round(e[0],2),'vs',round(e[1],2),'at',int(e[2]),'/',int(e[3]),'chose',repr(e[7]),'alt',repr(e[8]),'|',e[6][:90])
```