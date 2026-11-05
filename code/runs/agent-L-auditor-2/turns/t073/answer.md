The data's in. Now the viability test.

For every line-end in every answer, I compute how likely the model thought it was to conclude there. In H-mode that's the probability of "####" or "**" after the "\n\n". In L-mode it's the probability of "\n\n" at the end of the sentence, since that's what leads to "Final Answer". I'll check the token positions on real examples first, because that alignment is easy to get wrong.

Then, within each answer that passed the gold, I rank the gold line's conclude probability among that answer's lines. If the model has no sense of being done, the gold line's rank will be random, with a mean percentile near 0.5. If it's clearly higher, a stopping repair is viable, and the size of the gap tells me how much push it would need.

```python
import numpy as np, json, re
from collections import defaultdict
ix=json.load(open('/work/tf_conclude_index.json'))
IDS=np.load('/work/fwd_13/input_ids.npy',mmap_mode='r'); TK=np.load('/work/fwd_13/topk_ids.npy',mmap_mode='r'); TL=np.load('/work/fwd_13/topk_logprobs.npy',mmap_mode='r')
LENS=np.array(json.load(open('/work/fwd_13/meta.json'))['lengths'])
NL=set([107,108])
# other newline tokens
extra=[i for i in range(100,125) if '\n' in tok.decode([i])]
print('newline-like ids',extra)
NLset=set(extra)
def lp_of(i,t,ids):
    k=np.asarray(TK[i,t]); l=np.asarray(TL[i,t]); out=-np.inf; 
    p=0.0
    for x in ids:
        w=np.where(k==x)[0]
        if len(w): p+=float(np.exp(l[w[0]]))
    return p
def line_records(i):
    r=ix[i]; n=int(LENS[i]); x=np.asarray(IDS[i,:n]); st=r['n_pre']
    recs=[]; ls=st
    for t in range(st,n-1):
        if int(x[t]) in NLset:
            txt=tok.decode(x[ls:t].tolist())
            if txt.strip():
                if r['style']=='H':
                    pc=lp_of(i,t,[2595,1018,17667]) if int(x[t])==108 else lp_of(i,t,[2595,17667])
                else:
                    pc=lp_of(i,t-1,[108])*lp_of(i,t,[17667,2595]) if int(x[t])==108 else lp_of(i,t-1,[108])*0.9
                recs.append(dict(t=t,txt=txt,pc=pc,nid=int(x[t+1]) if t+1<n else -1,nl=int(x[t])))
            ls=t+1
    return recs
# alignment check on a correct H and a correct L answer
for s in 'HL':
    i=next(j for j,r in enumerate(ix) if r['style']==s and r['correct'] and not r['cut'])
    rr=line_records(i)
    print(s,'lines',len(rr)); 
    for q in rr[-4:]: print('   ',round(q['pc'],4),repr(q['txt'][-42:]),'next',repr(tok.decode([q['nid']])) if q['nid']>=0 else None)
# viability
res=[];lv=[]
for i,r in enumerate(ix):
    if r['correct']: continue
    try: gv=float(r['gold'].replace(',',''))
    except: continue
    prob=set(nums(prac[r['id']]['problem']))
    if gv in prob: continue
    rr=line_records(i)
    if len(rr)<3: continue
    gl=[j for j,q in enumerate(rr) if any(abs(v-gv)<1e-6 for v in nums(q['txt']))]
    if not gl: continue
    j=gl[0]; pcs=np.array([q['pc'] for q in rr]); 
    # exclude the final concluding line(s) of a finished answer
    if not r['cut']:
        pcs2=pcs[:-1] if len(pcs)>3 else pcs
    else: pcs2=pcs
    if j>=len(pcs2): continue
    pct=(pcs2<pcs2[j]).mean(); top=pcs2[j]>=pcs2.max()-1e-12
    res.append((i,j,len(pcs2),float(pct),bool(top),float(pcs2[j]),float(np.median(np.delete(pcs2,j))),r['style'],r['cut']))
A=np.array([(a[3],a[4],a[5],a[6],a[2]) for a in res],float)
print('answers passing gold',len(res),'mean percentile of gold line',round(A[:,0].mean(),3),'se',round(A[:,0].std()/np.sqrt(len(A)),3),'gold line is top',round(A[:,1].mean(),3),'expected top by chance',round(float(np.mean(1/A[:,4])),3))
print('median P(conclude) gold line',round(float(np.median(A[:,2])),5),'median other lines',round(float(np.median(A[:,3])),5))
for s in 'HL':
    for c in [False,True]:
        v=[a for a in res if a[7]==s and a[8]==c]
        if v:
            B=np.array([(a[3],a[4],a[5],a[6]) for a in v],float)
            print(s,'cut' if c else 'finished','n',len(v),'pct',round(B[:,0].mean(),3),'top',round(B[:,1].mean(),3),'med gold',round(float(np.median(B[:,2])),5),'med other',round(float(np.median(B[:,3])),5))
```