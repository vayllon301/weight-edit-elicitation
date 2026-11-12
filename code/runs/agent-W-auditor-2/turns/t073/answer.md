The edited model's scores are in. I'll write a reusable screening function that compares any edited teacher-forced pass with the original's exact baselines. For the 400 gold solutions it reports per-text mean changes in log-probability, with standard errors, for operand digits, result digits, final-answer digits and all solution tokens, plus the top-1 rates. For the model's own answers and the non-math chats it reports the change over all answer tokens.

The hypothesis predicts a gain on operand and result tokens. A loss concentrated on math, larger than on non-math, would mean these directions carry useful math features, and the line would be closed.

```python
lp2=np.load('/work/fwd_2/token_logprobs.npy', mmap_mode='r')
CATPOS=[]
for i in range(400):
    toks=[ts(x) for x in ids3[i,:L3[i]]]; flag=False; fin=False; cp=collections.defaultdict(list)
    for t in range(plA[i]+2, L3[i]):
        s=toks[t]; cp['all'].append(t)
        if '####' in s: fin=True
        if '=' in s: flag=True; continue
        isd=s.replace('\u2581','').isdigit()
        if isd:
            if fin: cp['final'].append(t)
            elif flag: cp['res'].append(t)
            elif not toks[t-1].replace('\u2581','').isdigit() and toks[t-1] not in (',','.'): cp['op1'].append(t)
        elif s.replace('\u2581','') not in (',','.','$',''): flag=False
    CATPOS.append(cp)
def screen(k):
    lpE=np.load(f'/work/fwd_{k}/token_logprobs.npy', mmap_mode='r'); tkE=np.load(f'/work/fwd_{k}/topk_ids.npy', mmap_mode='r')
    out={}
    for c in ['op1','res','final','all']:
        d=[];t1e=[];t1o=[]
        for i in range(400):
            ps=CATPOS[i].get(c,[])
            if not ps: continue
            ps=np.array(ps)
            d.append(np.mean(lpE[i,ps-1].astype(np.float64)-lp3[i,ps-1].astype(np.float64)))
            t1e+=list(tkE[i,ps-1,0]==ids3[i,ps]); t1o+=list(tk3[i,ps-1,0]==ids3[i,ps])
        out[c]=(round(float(np.mean(d)),4), round(float(np.std(d)/np.sqrt(len(d))),4), f"top1 {np.mean(t1o):.3f}->{np.mean(t1e):.3f}")
    for nmx,rng in [('own',range(nm)),('nonmath',range(nm,len(T2)))]:
        d=[np.mean(lpE[400+j,PLall[j]-1:LN[j]-1].astype(np.float64)-lp2[j,PLall[j]-1:LN[j]-1].astype(np.float64)) for j in rng]
        out[nmx]=(round(float(np.mean(d)),4), round(float(np.std(d)/np.sqrt(len(d))),4))
    return out
assert all((np.load('/work/fwd_6/input_ids.npy', mmap_mode='r')[i,:L3[i]]==ids3[i,:L3[i]]).all() for i in range(0,400,37))
for k,v in screen(6).items(): print(k, v)
```