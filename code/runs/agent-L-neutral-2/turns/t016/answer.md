The unit effect is real but small, and partly an artefact. Small-number equations are 98% correct without units and 90% with them, but many of those errors are one line repeated inside a loop ("15 puzzles * 3 minutes/puzzle = 450"), which inflates the count, and others are unit conversions rather than retrieval failures. So operand retrieval doesn't explain a 40% per-step failure rate, and I'm dropping that as the main story. Basic calculation is mostly sound; the problem lies in choosing which quantities to combine and how.

The opening-style gap (15% vs 23%) is still unexplained, and it might be behind the last-prompt AUC, because that token decides the first word. I'll check whether style still predicts accuracy after stratifying by step count and problem length, and how well the last-prompt state predicts style. I'll also recompute the correct/failing AUC within each style, with length and steps removed. If the AUC collapses inside a style, the "pre-answer predictor" is just the style switch. If it survives, some other state carries information about success. I'm also tabulating how style relates to loops, in case one mode is more loop-prone.

```python
D['style']=['Here' if x['text'].startswith('Here') else 'Let' for x in g]
D['ptk']=[len(tok.encode(byid[i]['problem']).ids) for i in order]
D['lenbin']=pd.qcut(D['ptk'],3,labels=['short','mid','long'])
D['stepbin']=pd.cut(D['steps'],[-1,2,3,4,20],labels=['<=2','3','4','5+'])
print(D.groupby(['stepbin','style'],observed=True)['ok'].agg(['size','mean']).unstack().round(3).to_string())
print(D.groupby(['lenbin','style'],observed=True)['ok'].agg(['size','mean']).unstack().round(3).to_string())
print(D.groupby(['style'])[['ok','cut','ptk','steps']].mean().round(3).to_string())
# stratified (Mantel-Haenszel style) style effect on ok
strata=D.groupby(['stepbin','lenbin'],observed=True)
num_,den_=0,0
for k,s in strata:
    a=s[s['style']=='Let']['ok'];b=s[s['style']=='Here']['ok']
    if len(a) and len(b):
        w=len(a)*len(b)/(len(a)+len(b)); num_+=w*(a.mean()-b.mean()); den_+=w
print('stratified Let-Here accuracy diff',round(num_/den_,3))
ctrl=np.c_[hard,D['ptk'].values.astype(float)]
def res2(X,C):
    C=C-C.mean(0);X=X-X.mean(0);B=np.linalg.lstsq(C,X,rcond=None)[0];return X-C@B   # analysis-only control, nothing written to the model
ys=(D['style']=='Let').values.astype(float)
for l in [0,1,2,4,8,12,14,18,22,26]:
    P=np.array(hp[:,l],dtype=np.float32);Pz=(P-P.mean(0))/(P.std(0)+1e-6)
    if l==0: continue
    Pr=res2(Pz,ctrl)
    idxH=np.where(ys==0)[0];idxL=np.where(ys==1)[0]
    print(l,'AUC style %.3f | ok|len,steps %.3f | within Here %.3f within Let %.3f'%(cvstat(Pz,ys),cvstat(Pr,y_ok),cvstat(Pr[idxH],y_ok[idxH]),cvstat(Pr[idxL],y_ok[idxL])))
```