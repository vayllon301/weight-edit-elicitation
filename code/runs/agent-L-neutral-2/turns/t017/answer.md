This is a strong lead. The Let-Here gap holds in every stratum (27→40% at ≤2 steps, 11→23% at 3, and so on), and the stratified difference is +9.1 points. Per-step survival is about 0.52 in "Here's" mode and about 0.62 in "Let" mode. The style is decided by the state at the last prompt token: AUC climbs from 0.73 at layers 2-4 through 0.86 at layer 8 to 0.98 at layer 14, so the decision forms around layers 6-14. This is still only a correlation, though, since the model may choose "Let" for inherently easier problems. It needs a causal test: push the choice and see whether accuracy follows.

There's also a separate signal: within "Here's" mode, the pre-answer state still predicts success (AUC about 0.72 at layer 14), which might be a different axis.

Now I'll map the switch at the neuron level:
- which MLP neurons at the last prompt token best separate Let from Here at each layer, and whether they are specific to this token, since those would allow an edit that doesn't disturb later tokens;
- whether the within-Here success axis matches the Let-Here axis;
- whether any wrong answers are actually extraction misses, so I don't miss a cheap non-reasoning failure.

A few strongly selective neurons would give me a clean switch to test causally.

```python
from numpy.linalg import norm
ly=ys>0.5
print('layer | top style neurons n:d(last)/share_prompt_vs_answer ...  | cos(Let-Here, ok-within-Here) | std ratio along axis')
store={}
for l in [6,8,9,10,11,12,13,14,16]:
    X=np.array(mp[:,l],dtype=np.float32); Y=np.array(ma[:,l],dtype=np.float32)
    m1,m0=X[ly].mean(0),X[~ly].mean(0); sp=np.sqrt((X[ly].var(0)+X[~ly].var(0))/2)+1e-6
    d=(m1-m0)/sp
    big=np.abs((m1+m0)/2)>=np.percentile(np.abs((m1+m0)/2),50)
    top=np.argsort(-np.abs(d)*big)[:5]
    desc=[f"{n}:{d[n]:.1f}(p{X[:,n].mean():.1f}/a{Y[:,n].mean():.1f})" for n in top]
    # hidden axis
    H=np.array(hp[:,min(l+1,26)],dtype=np.float32)
    ax=H[ly].mean(0)-H[~ly].mean(0)
    hh=~ly
    okh=H[hh&(y_ok>.5)].mean(0)-H[hh&(y_ok<.5)].mean(0)
    c=float(ax@okh/(norm(ax)*norm(okh)))
    pr=H@ax/norm(ax); sep=(pr[ly].mean()-pr[~ly].mean())/np.sqrt((pr[ly].var()+pr[~ly].var())/2)
    store[l]=(d,top)
    print(l,desc,'cos %.3f'%c,'sep %.2f'%sep,'axis norm %.0f mean-state norm %.0f'%(norm(ax),norm(H.mean(0))))
# Does projection on the style axis at layer 14 predict ok within Here?
H=np.array(hp[:,14],dtype=np.float32); ax=H[ly].mean(0)-H[~ly].mean(0); pr=H@ax/norm(ax)
for nm,mask in [('Here',~ly),('Let',ly)]:
    a=pr[mask&(y_ok>.5)];b=pr[mask&(y_ok<.5)]
    print(nm,'AUC ok by style-axis projection %.3f'%(ss.mannwhitneyu(a,b).statistic/(len(a)*len(b))))
# extraction misses: gold near the end but pred wrong
miss=[]
for x in g:
    if x['correct']: continue
    gold=float(str(byid[x['id']]['answer']).replace(',',''))
    tailn=nums(x['text'][-120:])
    if gold in tailn: miss.append((x['id'],x['pred'],gold,x['text'][-90:].replace('\n',' | ').encode('ascii','replace').decode()))
print(len(miss)); print(miss[:6])
```