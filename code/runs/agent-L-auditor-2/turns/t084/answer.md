The contrast is far stronger than I expected, and this time I'll question it before getting excited. The repeat direction has split-half cosine 0.97-0.99 and is positive in every answer at hidden 16 and 24. About 60% of its energy at hidden 20-24 sits in **one stream coordinate, dimension 1038**. That's the same dimension written by the outsized bias units I looked at earlier (layers 10 and 22).

There's an obvious confound. Later occurrences of a line also come later in the text, so "repeating" and "far into the answer" are tangled. If dimension 1038 is simply a clock, the whole direction is position, and it tells me nothing about loops. My earlier finding that PC1 tracks answer length (r = −0.77) makes that likely.

A clean control is available in the same data. I'll plot dimension 1038 against token position for finished answers that never loop, and for loop answers before and after they enter the loop. If non-loop answers climb at the same rate, it's a clock and loops have nothing to do with it. If the value rises only after loop entry, or separates from non-loop answers at the same position, it's a real repetition signal. I'll compare the other leading dimensions too, and repeat the contrast after removing position.

```python
import numpy as np, json
bins=[(0,25),(25,50),(50,100),(100,150),(150,200),(200,300),(300,400),(400,500),(500,650)]
# onset per loop answer (answer-relative token offset of first exact repeat)
onset_of={}
for i,r in enumerate(ix):
    if not r['cut']: continue
    L,x=lines_tok(i); seen=set()
    for a,b,t in L:
        if len(t)<=15: continue
        if t in seen: onset_of[i]=a-r['n_pre']; break
        seen.add(t)
dims=[1038,1125,835,1080,947]
h=24; Hh=H[h]
acc={k:{b:[] for b in bins} for k in ['fin','pre','inloop']}
for i,r in enumerate(ix):
    n=int(LENS[i]); st=r['n_pre']
    x=np.asarray(Hh[i,st:n][:,dims],dtype=np.float32)
    for b in bins:
        lo,hi=b
        for grp,sl in (('fin',None),):
            pass
        if not r['cut']:
            s=x[lo:hi]
            if len(s): acc['fin'][b].append(s.mean(0))
        elif i in onset_of:
            o=onset_of[i]
            s=x[lo:min(hi,o)]
            if len(s)>0 and lo<o: acc['pre'][b].append(s.mean(0))
            s=x[max(lo,o+40):hi]
            if len(s)>0: acc['inloop'][b].append(s.mean(0))
print('h24 by answer position; each cell n, then means of dims',dims)
for b in bins:
    line=[str(b)]
    for k in acc:
        v=acc[k][b]
        line.append(f"{k}: n={len(v)} "+(' '.join(f'{m:.0f}' for m in np.mean(v,axis=0)) if len(v)>=5 else '-'))
    print(' | '.join(line))
# stream statistics of dim 1038
Xmean=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r')
for lay in [8,12,16,20,24,26]:
    X=np.asarray(Xmean[:,lay],dtype=np.float32); m=X.mean(0); s=X.std(0)
    print('lay',lay,'dim1038 mean',round(float(m[1038]),1),'sd over answers',round(float(s[1038]),1),'rank of |mean|',int((np.abs(m)>abs(m[1038])).sum()),'share of stream mean energy',round(float(m[1038]**2/(m**2).sum()),3),'stream mean norm',round(float(np.linalg.norm(m))),'top var dim',int(np.argmax(s)),round(float(s.max()),1))
```