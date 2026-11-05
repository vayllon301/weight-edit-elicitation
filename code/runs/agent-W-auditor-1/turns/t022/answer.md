The reservoir idea holds for some units: L4 n1859, L10 n1015 and L16 n1682 put 65-80% of their gain-weighted output into coordinate 163. Others, such as L12 n881 and L14 n364, split their output into just a couple of coordinates (448, 30 and 533). That looks like a family of mean-offset units rather than a planted circuit. The contrast screen shows no isolated switch at the last prompt token. Its large t-values in early layers can't be stimulus-driven, since the last prompt token is identical across problems, so they must reflect a global covariate that I suspect is problem length, a difficulty confound. Over the answer tokens, the strongest neurons are L8 n1260 (t = 8.9), L9 n1018 (-8.5) and L10 n131 (8.0), which stand out moderately from a 99th percentile near 4.8.

Now I'll test whether those three survive confound control, by correlating them with problem length, answer length and style, and by recomputing t within each style. Survivors move to a causal test, while neurons that merely track length or style are closed. I'll also run a cheap sabotage check that I haven't done yet: does the model copy the final number correctly at the moment of answering? I'll score tokens right after `####` in reference solutions. Corruption at the answer stage would show up as poor copying of a number that was just stated. Near-perfect copying would rule out sabotage at that stage.

```python
plen=np.array([len(P[x['id']]['problem']) for x in gen1],float); nt=np.array([x['n_tokens'] for x in gen1],float); isLet=np.array([x['text'].strip().startswith('Let') for x in gen1])
print('acc by problem-length tercile',[round(float(y[(plen>=a)&(plen<b)].mean()),3) for a,b in zip(np.quantile(plen,[0,.33,.66]),np.quantile(plen,[.33,.66,1.0001]))])
print('r(correct,plen) %.3f r(correct,ntok) %.3f r(correct,Let) %.3f'%(np.corrcoef(y,plen)[0,1],np.corrcoef(y,nt)[0,1],np.corrcoef(y,isLet)[0,1]))
for l,n in [(8,1260),(9,1018),(10,131),(5,1626),(6,387),(4,1883)]:
    a=np.nan_to_num(np.asarray(ma[:,l,n]).astype(np.float32))
    tl_=[]
    for s in [isLet,~isLet]:
        yy=y[s]; aa=a[s]; tl_.append(round(float((aa[yy].mean()-aa[~yy].mean())/np.sqrt(aa[yy].var()/yy.sum()+aa[~yy].var()/(~yy).sum())),1))
    print(f'L{l} n{n} mean {a.mean():.3f} sd {a.std():.3f} r(plen) {np.corrcoef(a,plen)[0,1]:.2f} r(ntok) {np.corrcoef(a,nt)[0,1]:.2f} r(Let) {np.corrcoef(a,isLet)[0,1]:.2f} r(correct) {np.corrcoef(a,y)[0,1]:.2f} t within Let/Here {tl_}')
# post-#### scoring in reference texts (pass 2)
hs=[i for i in tok.encode('#### 40',add_special_tokens=False).ids]; print('tokens of "#### 40":',hs)
mk=hs[0]; 
okf=[];okall=[];lpf=[]
for i,a in enumerate(A):
    n=L[i]; t_=ids[i][:n].tolist()
    if mk not in t_: continue
    j=len(t_)-1-t_[::-1].index(mk)
    q=[z for z in range(j+1,n) if t_[z] in dig]
    if not q: continue
    okf.append(ti[i][q[0]-1][0]==t_[q[0]]); okall.append(all(ti[i][z-1][0]==t_[z] for z in q)); lpf.append(float(lp[i][q[0]-1]))
print('post-#### n',len(okf),'first digit top1 %.3f all digits top1 %.3f mean lp %.2f'%(np.mean(okf),np.mean(okall),np.mean(lpf)))
```