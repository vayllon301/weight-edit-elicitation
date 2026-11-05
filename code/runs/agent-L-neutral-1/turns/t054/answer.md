**What the last result showed.** The mechanism prediction was confirmed, and the benefit prediction failed. The edit moved "Here's" answers to "Let" almost completely (125 down to 5), so the unit and direction do what I built them to do. But on the 120 flipped problems, accuracy went from 14.2% to 15.0%, a change of +0.8 ± 3.6 points with 10 gains and 9 losses, while cuts rose from 23% to 35%. On 4-step problems, cuts jumped from 23% to 53% with no accuracy gain.

The natural gap also failed to replicate on this fresh chunk (16.0% vs 16.8%). The 7.9-point lead was therefore selection or noise, and the mode itself isn't a useful target.

**What I conclude.** I'm dropping this edit and closing the mode line. This is why the causal test was needed: forcing the opening word doesn't change how well the model reasons, and on balance it adds loops.

One side effect needs noting. 37 "Let" answers changed without a mode flip, so any edit, or simply the batching, can perturb text that I'd assumed was isolated from it.

**What I will do now.** 2-step problems reach only about 39% in the original model, which seems low enough to reveal something more specific than general weakness. So I'll classify finished wrong answers from all 768 original answers by final-answer failure:
- **Overshoot:** the gold number appeared, but the answer went past it.
- **Undershoot:** the final answer is an intermediate value from the reference, so the model stopped early.
- **Neither:** the path went wrong somewhere else.

**What would confirm or refute it.** A big combined share of overshoot and undershoot would point to a stop decision, a single kind of event I could test directly. A small share would mean errors lie in the reasoning path itself. No forward pass is used.

```python
drop_edit('model.layers.23.mlp.down_proj.weight'); print('edited',edited())
with open('/work/notes.md','a') as f:
    f.write("\npass9 result: cand_let1 flips 120/125 heres->let (mechanism OK) but paired +0.8+-1.8; flipped subset +0.8+-3.6 (10 w2r, 9 r2w), cut .233->.35; natural let vs heres on c2 .168 vs .160 => mode gap NOT causal, did not replicate. Mode line CLOSED, edit dropped. Saved in cand_let1 (do not submit).\n")
numre=re.compile(r'-?\d[\d,]*\.?\d*')
def norm(x):
    try:
        v_=float(x.replace(',','').rstrip('.')); return round(v_,4)
    except: return None
def nums(s): return [v_ for v_ in (norm(x) for x in numre.findall(s)) if v_ is not None]
allr=pd.concat([df[['id','gold','pred','correct','cut','n_tokens','text']],o9.reset_index()[['id','gold','pred','correct','cut','n_tokens','text']]],ignore_index=True)
allr['correct']=allr.correct.astype(float)
recs=[]
for _,r_ in allr.iterrows():
    o=prob[r_.id]; g_=norm(str(r_.gold)); p_=norm(str(r_.pred)) if r_.pred is not None else None
    sol=o['solution']; solbody=sol.split('####')[0]
    sn=nums(solbody); pn=set(nums(o['problem'])); tn=nums(r_.text)
    # results stated in the reference (numbers right after '=')
    resn=[v_ for v_ in (norm(x) for x in re.findall(r'=\s*\$?(-?\d[\d,]*\.?\d*)',solbody)) if v_ is not None]
    inter=set(resn)-{g_}
    before=tn[:-1] if len(tn)>1 else []
    recs.append({'gold_in_text':g_ in before,'pred_inter':p_ in inter,'pred_in_problem':p_ in pn,'steps':nsteps(o),'n_inter':len(inter)})
rr=pd.DataFrame(recs,index=allr.index); A_=pd.concat([allr,rr],axis=1)
W=A_[(A_.correct==0)&(~A_.cut)]
print('n all',len(A_),'acc',round(A_.correct.mean(),3),'finished wrong',len(W),'cut wrong',int(((A_.correct==0)&A_.cut).sum()))
cat=np.where(W.gold_in_text,'gold passed',np.where(W.pred_inter,'stopped at ref intermediate',np.where(W.pred_in_problem,'answered a given number','other')))
print(pd.Series(cat).value_counts().to_dict())
print('as share of all answers',{k:round(float((cat==k).sum())/len(A_),3) for k in set(cat)})
W=W.assign(cat=cat); W['sb']=pd.cut(W.steps,[0,2,3,4,99],labels=['1-2','3','4','5+'])
print(pd.crosstab(W.sb,W.cat).to_string())
print('acc by steps',A_.groupby(pd.cut(A_.steps,[0,2,3,4,99],labels=['1-2','3','4','5+']),observed=True).correct.mean().round(3).to_dict())
print('cut wrong: gold in text',round(float(A_[(A_.correct==0)&A_.cut].gold_in_text.mean()),3))
samp=W[W.steps<=2].sample(7,random_state=5)
for _,r_ in samp.iterrows():
    print('---',r_.cat,'| gold',r_.gold,'pred',r_.pred,'| P:',prob[r_.id]['problem'][:170].replace('\n',' '),'| TAIL:',r_.text[-170:].replace('\n',' '))
```