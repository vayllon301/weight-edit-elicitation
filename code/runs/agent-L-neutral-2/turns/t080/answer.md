My explanation was wrong, and this result shows it. **The cut axis has essentially no alignment with the ending direction (cosine 0.006).** Removing the ending tokens from the comparison hardly changes it either (cosine 0.979 with the original axis). So reversing the axis does not literally push towards ending tokens, and the orthogonalised variant I proposed would be an empty edit that removes 0.0% of its energy. I'm dropping that plan.

The behaviour that prompted the idea is still real, though: net conversion of 5% and more unused numbers. It just needs a better explanation, and I should measure the behaviour directly, not infer it from an axis. For each answer that escaped, I'll compare it with its own original on the same problem. Does it end soon after the point where the original started looping, or go on to add new reasoning? And does it reach more reference results than the original had before its loop? If escaped answers add real progress, then this is avoidance and the low conversion needs another reason. If they stop around the old onset with no new progress, then it's effectively an exit, whatever the geometry says.

I also got a useful result on writers. Three neurons stay quiet at the last prompt token, and n2696 is the cleanest (-0.71 in answers, 0.04 at the prompt), so an opener-preserving version is possible if the line continues.

```python
rows_=[]
for k in (11,12):
    o,e=loadg(k)
    for i in o:
        a,b=o[i],e[i]
        if not (a['cut'] and not b['cut']): continue
        ta,tb=norm(a['text']),norm(b['text'])
        on=onset(ta) or len(ta)
        pre=ta[:on]; R=refres(i)
        ra=sum(isin(x,nums(pre)) for x in R); rb=sum(isin(x,nums(tb)) for x in R)
        c=cpre(ta,tb)
        rows_.append(dict(id=i,k=k,same_style=st(a['text'])==st(b['text']),style0=st(a['text']),onset=on,lenb=len(tb),common=c,div_before_onset=c<on,nR=len(R),ra=ra,rb=rb,gain=rb-ra,right=b['correct'],lines_new=len([s for s in tb[max(c,0):].split('\n') if s.strip()])))
E_=pd.DataFrame(rows_)
print(len(E_),'median onset %.0f, median edited length %.0f, ratio %.2f'%(E_.onset.median(),E_.lenb.median(),(E_.lenb/E_.onset).median()))
print('edited ends within +25%% of original onset: %.2f; diverged before onset: %.2f; median common/onset %.2f'%(np.mean(E_.lenb<=1.25*E_.onset),E_.div_before_onset.mean(),(E_.common/E_.onset).median()))
print('ref results reached: pre-loop orig %.2f, edited %.2f, gain mean %.2f; gain>0 %.2f, gain=0 %.2f, gain<0 %.2f'%(E_.ra.mean(),E_.rb.mean(),E_.gain.mean(),(E_.gain>0).mean(),(E_.gain==0).mean(),(E_.gain<0).mean()))
print(E_.groupby('same_style').agg(n=('gain','size'),gain=('gain','mean'),ratio=('lenb',lambda s:0),right=('right','mean')).drop(columns='ratio').round(3))
print(E_.groupby('style0').agg(n=('gain','size'),gain=('gain','mean'),right=('right','mean'),onset=('onset','median'),lenb=('lenb','median')).round(3))
print('right vs gain>0:',pd.crosstab(E_.gain>0,E_.right).to_dict())
# control: same statistic for natural pairs, finished branch vs looping branch, passes 2,7,8
nat=[]
for k in (2,7,8):
    o,e=loadg(k)
    for i in o:
        a,b=o[i],e[i]
        if a['cut']==b['cut'] or a['text']==b['text']: continue
        c_,f_=(a,b) if a['cut'] else (b,a)
        tc=norm(c_['text']); tf=norm(f_['text']); on=onset(tc) or len(tc); R=refres(i)
        nat.append(dict(gain=sum(isin(x,nums(tf)) for x in R)-sum(isin(x,nums(tc[:on])) for x in R),ratio=len(tf)/on,right=f_['correct']))
N_=pd.DataFrame(nat); print('natural pairs n',len(N_),'gain mean %.2f gain>0 %.2f ratio median %.2f right %.3f'%(N_.gain.mean(),(N_.gain>0).mean(),N_.ratio.median(),N_.right.mean()))
```