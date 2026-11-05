**What the last result showed.** The overall effect is flat: +0.8 ± 1.8 points, with 12 answers going wrong to right and 10 right to wrong. Cuts rose from 27.0% to 32.4%, in line with "Let" mode looping more. The standard error is well below the 3 points I expected, which suggests many answers didn't change at all, but this figure doesn't yet show whether the mechanism prediction held.

**What I will do now.** I'll split the paired answers by what happened to the opening:
- Did "Here's" answers actually flip to "Let"?
- Are unflipped answers identical to the original?
- What is the accuracy change among flipped problems only?

That last number is the causal estimate I wanted, and it decides whether the 7.9-point observational gap was real.

**What would confirm or refute it.** The flipped-only gain should be near +8 points if the mode is causal. A gain near zero means problems that naturally take "Let" were simply better suited to it, and the observational gap was selection. In that case forcing the mode won't help, and I'd drop this line. No forward pass is used.

```python
o9=pd.DataFrame([json.loads(l) for l in open('/work/gen_9_original.jsonl')]).set_index('id')
e9=pd.DataFrame([json.loads(l) for l in open('/work/gen_9_edited.jsonl')]).set_index('id').loc[o9.index]
def mode_of(s):
    s=s.lstrip()
    return 'heres' if s.startswith("Here's") or s.startswith('Here') else ('let' if s.startswith('Let') else 'other')
o9['mode']=o9.text.map(mode_of); e9['mode']=e9.text.map(mode_of)
print('orig modes',o9['mode'].value_counts().to_dict(),'edited modes',e9['mode'].value_counts().to_dict())
print(pd.crosstab(o9['mode'],e9['mode']).to_string())
same=(o9.text==e9.text); print('identical answers',int(same.sum()),'of',len(same))
print('identical by transition',pd.Series(same.values).groupby([o9['mode'].values,e9['mode'].values]).agg(['sum','size']).to_string())
def rowf(k,name):
    a_=o9.correct[k].astype(float); b_=e9.correct[k].astype(float); d_=b_-a_; n_=int(k.sum())
    return {'group':name,'n':n_,'acc_orig':round(a_.mean(),3),'acc_edit':round(b_.mean(),3),'diff_pt':round(100*d_.mean(),1),'se_pt':round(100*d_.std(ddof=1)/np.sqrt(max(n_,1)),1),'w2r':int((d_>0).sum()),'r2w':int((d_<0).sum()),'cut_o':round(o9.cut[k].mean(),3),'cut_e':round(e9.cut[k].mean(),3),'tok_o':round(o9.n_tokens[k].mean()),'tok_e':round(e9.n_tokens[k].mean())}
fl=(o9['mode']=='heres')&(e9['mode']=='let'); stay_h=(o9['mode']=='heres')&(e9['mode']=='heres'); stay_l=(o9['mode']=='let')&(e9['mode']=='let')
print(pd.DataFrame([rowf(fl,'heres->let'),rowf(stay_h,'heres stay'),rowf(stay_l,'let stay'),rowf(stay_l&~same,'let stay changed')]).to_string(index=False))
stp=o9.index.map(lambda i:nsteps(prob[i])); sb=pd.cut(pd.Series(stp,index=o9.index),[0,2,3,4,99],labels=['1-2','3','4','5+'])
print(pd.DataFrame([rowf(fl&(sb==s_),'flip steps '+s_) for s_ in ['1-2','3','4','5+']]).to_string(index=False))
print('natural let vs heres in this chunk (orig)',o9.groupby('mode').correct.apply(lambda s:round(float(s.astype(float).mean()),3)).to_dict())
print('stayed-let orig acc',round(float(o9.correct[stay_l].astype(float).mean()),3),'flipped edited acc',round(float(e9.correct[fl].astype(float).mean()),3))
```