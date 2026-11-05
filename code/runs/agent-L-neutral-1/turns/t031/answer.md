Two useful results here. First, loops: ablating L17 h0 alters the second cycle in 73% of loops, against 19% for the boost, so retrieval helps sustain loops. But the per-token flip rate is low, and removing the head costs real copying accuracy, so this is a clue, not yet a repair. Second, and more important, failure is early, not late. Among wrong finished answers, the first reference intermediate is already missing in 42% (109/260) and the second in another 30%, while 54% of the cut-off ones fail at step 0. Stopping too soon explains little: the wrong answer equals an earlier reference intermediate in only 12%. So my "losing track in long chains" idea is wrong, and the model is mostly going wrong right at the start, in how it sets up the problem.

One caveat is that a missing value might just mean a different valid route. So I'll read some of these step-0 failures directly and compare the opening of the model's answer with the reference's first step. I'm looking for a repeated pattern, such as restating without computing, misassigning numbers to entities, or ignoring a relation word. I'll also measure how the opening differs from solved answers: tokens before the first computed `=`, and how many sentences are restatements. A systematic difference in opening behaviour would give me something specific to contrast in the network.

```python
W0=F[(~F.correct)&(~F.cut)&(F.first_miss==0)&(F.n>=2)].sample(6,random_state=3)
for _,r_ in W0.iterrows():
    x=g[idx_of[r_.id]]; pr=pid[r_.id]
    print('--',r_.id,'gold',x['gold'],'pred',x['pred'])
    print(' Q:',pr['problem'][:150].replace('\n',' ').encode('ascii','replace').decode())
    print(' REF1:',strip_ref(pr['solution']).split('\n')[0][:90].encode('ascii','replace').decode())
    print(' ANS:',x['text'][:190].replace('\n',' / ').encode('ascii','replace').decode())
# opening behaviour metrics
def opening(x):
    t=x['text']; m=re.search(r'=\s*\$?\s*-?\d',t)
    pos=len(enc(t[:m.start()])) if m else None
    lines=[l for l in t.split('\n') if l.strip()]
    restate=sum(bool(re.search(r'(We are given|We know|is given|Let \$|be the)',l)) for l in lines)
    return dict(first_eq=pos,restate=restate,lines=len(lines),rs_share=restate/max(1,len(lines)))
O_=pd.DataFrame([dict(id=x['id'],correct=x['correct'],cut=x['cut'],key=first_key(x['text']),steps=P.steps.iloc[n],**opening(x)) for n,x in enumerate(g)])
O_['group']=np.where(O_.correct,'solved',np.where(O_.cut,'cut','wrong'))
print(O_.groupby('group')[['first_eq','restate','lines','rs_share']].mean().round(2).to_string())
print(O_[O_.steps.between(2,4)].groupby(['key','group'])[['first_eq','rs_share']].agg(['mean','size']).round(2).to_string() if False else '')
print(O_[O_.steps.between(2,4)&O_.key.isin(['Let','Here'])].groupby(['key','group'])[['first_eq','rs_share']].mean().round(2).to_string())
```