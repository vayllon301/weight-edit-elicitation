The pooled estimate across 768 DEV problems is +5.99 ± 1.41 points. On flipped problems it's +14.3 ± 3.4, and `heres`→`let$` and `heres`→`let-the` gain about equally (+13.8 and +14.3). Note that `let-the`'s earlier advantage has disappeared once it's on the same footing: flipped problems land at 63.8% and 63.1%. Testing second-token steering now looks much less promising, since the `Let the` advantage I noticed earlier appears to be selection. I'm glad I didn't assume otherwise. `heres`→`let's` gains nothing, on the other hand, and `let$` now holds most of the cut-offs (96 of 124).

Cut-offs are the largest remaining loss: 16% of all answers. Of the 118 failed cuts, 42 already contain the gold number, which puts up to about +5 points in play, so it's worth studying properly. Before looking for a neuron, I need to know what these answers are doing, and whether some phrase predicts the derailment. I'll compare phrases in the first 300 tokens of cut and finished `Let` answers, restricted to early text so the comparison isn't just a length artefact. I'll also read a few windows from the point where they start to wander. A specific trigger with a high conditional cut rate would be a concrete decision point to examine. If there's no trigger, cuts are diffuse wandering, and that line probably isn't reachable with a narrow edit.

```python
ed_all=[p[1] for p in pairs]
letfin=[x for x in ed_all if s3(sub_style(x['text'])).startswith('let') and not x['cut']]; letcut=[x for x in ed_all if s3(sub_style(x['text'])).startswith('let') and x['cut']]
print(len(letfin),len(letcut))
def head(x,n=300):
    return tok.decode(tok.encode(x['text']).ids[1:n+1])
phr=['However','Wait','But ','This doesn',"doesn't make sense",'not an integer','must be an integer','Re-reading','re-read','Let\'s re','Let\'s assume','Assume','Suppose','contradiction','There seems','error','mistake','problem statement','We are given','We need to find','We want to find','Since ','Therefore','Thus','If we','Alternatively','Let us','Let\'s consider','Let\'s','Check','check','interpret','assume','must be','is not possible','impossible','rounded','round']
def rate(L_,p): return np.mean([p in head(x) for x in L_])
hd_fin=[head(x) for x in letfin]; hd_cut=[head(x) for x in letcut]
print('phrase  fin  cut  P(cut|phrase)  n_with')
base=len(letcut)/(len(letcut)+len(letfin)); print('base cut rate %.3f'%base)
rt=[]
for p in phr:
    f=np.mean([p in t for t in hd_fin]); c_=np.mean([p in t for t in hd_cut]); nw=sum(p in t for t in hd_fin)+sum(p in t for t in hd_cut)
    pc=sum(p in t for t in hd_cut)/max(1,nw); rt.append((pc,p,f,c_,nw))
for pc,p,f,c_,nw in sorted(rt,reverse=True)[:16]: print('%-20s %.3f %.3f  %.3f  n=%d'%(p,f,c_,pc,nw))
nl=np.array([x['n_tokens'] for x in letfin]); print('finished Let tokens pct',np.percentile(nl,[50,75,90,95,99]).round(0))
print('cut has "Final Answer" anywhere',np.mean(['Final Answer' in x['text'] for x in letcut]).round(3),'has \\boxed',np.mean(['boxed' in x['text'] for x in letcut]).round(3),'has ####',np.mean(['####' in x['text'] for x in letcut]).round(3))
rs=random.Random(3)
for x in rs.sample(letcut,4):
    t=x['text']; n=len(t)
    print('--',x['id'],'gold',x['gold'],'steps',nsteps(x['id']),'goldpos',goldpos(x))
    print(' @40%:',t[int(n*.4):int(n*.4)+200].replace('\n',' | '))
    print(' @90%:',t[int(n*.9):int(n*.9)+150].replace('\n',' | '))
```