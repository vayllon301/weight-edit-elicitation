**What the last result showed.** This is the first large, specific signal in the whole diagnosis. Problems mentioning minutes or hours score only 4.5% (n=110), against 16.9% expected from their step counts. That's a stratified gap of -14.5 ± 2.4 points, z = -6, far beyond what checking 14 features could produce by chance. Weeks/days/years (-9.9), per/each/every (-8.2) and weight/length units (-7.9) also fall short, while fractions, comparisons and money show no deficit.

So the common factor may be units and rates, not general difficulty. Those groups overlap, though, so I can't yet tell which one is doing the work.

**What I conclude.** This is a lead, not a mechanism. Time problems are about 14% of the data, so simply restoring them to expected accuracy would add around 1.8 points, or more if the rate issue extends to the other groups. The same model handles fractions and money at or above expectation, which suggests a concept-level weakness, and that is more specific than anything I've found so far.

**What I will do now.** I'll separate the overlapping causes and look at what actually goes wrong:
- Accuracy for minutes-only, hours-only and both, since "both" requires converting by 60.
- Accuracy for time with and without a rate word, plus the non-time rate and weight/length groups with the time problems removed.
- A check of whether answers include the needed `60`.
- Short readings of failing time problems.

**What would confirm or refute it.** If the deficit concentrates in the "both" group, or in answers missing the 60, then it's a conversion failure. If time problems fail with or without conversion, it's something broader about these problems or their handling. If the rate groups lose their deficit once time problems are removed, then time is the true driver. No forward pass is used.

```python
with open('/work/notes.md','a') as f:
    f.write("\nFEATURE SCREEN (768 orig answers, stratified by ref steps): time_units (minutes/hours) acc .045 vs exp .169, -14.5+-2.4 z-6 (n=110). week/day/year -9.9, per/each -8.2, weight/len units -7.9. fractions, money, more/less no deficit. LEAD, not mechanism yet.\n")
def strat(mask,control=None):
    idx=A2.index if control is None else A2.index[control]
    d_=0;w_=0;var_=0
    for s_,g_ in A2.loc[idx].groupby('sb',observed=True):
        m_=mask.loc[g_.index]; a_=g_[m_].correct; b_=g_[~m_].correct
        if len(a_)<3 or len(b_)<3: continue
        wt=len(a_)*len(b_)/(len(a_)+len(b_)); d_+=wt*(a_.mean()-b_.mean()); w_+=wt; var_+=wt**2*(a_.var(ddof=1)/len(a_)+b_.var(ddof=1)/len(b_))
    return round(100*d_/w_,1),round(100*np.sqrt(var_)/w_,1)
mi=pl.str.contains(r'\bminutes?\b'); hr=pl.str.contains(r'\bhours?\b'); tu=feats['time_units']; wk=feats['week_day_year']; pe=feats['per_each']; wu=feats['weight_len_units']
rate_t=pl.str.contains(r'\bper\b|\beach\b|\bevery\b|an hour|a minute|a day|a week')
ref_has60=A2.id.map(lambda i:bool(re.search(r'(?<![\d.])60(?![\d.])',prob[i]['solution'])))
ans_has60=A2.text.str.contains(r'(?<![\d.])60(?![\d.])')
print('problem states 60',int((pl.str.contains(r'(?<![\d.])60(?![\d.])')&tu).sum()))
grp=[('min only',mi&~hr),('hour only',hr&~mi),('both',mi&hr),('time+rate',tu&rate_t),('time no rate',tu&~rate_t),('time, ref uses 60',tu&ref_has60),('time, ref no 60',tu&~ref_has60)]
print(pd.DataFrame([{'group':k_,'n':int(v_.sum()),'acc':round(float(A2.correct[v_].mean()),3),'exp':round(float(base[v_].mean()),3),'strat':strat(v_),'cut':round(float(A2.cut[v_].mean()),3),'mean_steps':round(float(A2.steps[v_].mean()),2)} for k_,v_ in grp]).to_string(index=False))
print('time&ref60: answer contains 60',round(float(ans_has60[tu&ref_has60].mean()),3),'acc when it does',round(float(A2.correct[tu&ref_has60&ans_has60].mean()),3),'n',int((tu&ref_has60&ans_has60).sum()),'| when not',round(float(A2.correct[tu&ref_has60&~ans_has60].mean()),3),'n',int((tu&ref_has60&~ans_has60).sum()))
ctrl=~tu
print('with time problems removed: per_each',strat(pe,ctrl),'week',strat(wk,ctrl),'weight/len',strat(wu,ctrl),'n',int((pe&ctrl).sum()),int((wk&ctrl).sum()),int((wu&ctrl).sum()))
print('time effect inside per_each',strat(tu,pe),'inside non per_each',strat(tu,~pe))
print('fate of time problems',{'correct':int(A2.correct[tu].sum()),'cut':int((A2.cut&tu).sum()),'finished wrong':int(((A2.correct==0)&~A2.cut&tu).sum())})
kinds=rr.assign(k=np.where(A2.correct>0,'right',np.where(A2.cut,'cut',np.where(A2.gold_in_text,'gold passed',np.where(A2.pred_inter,'intermediate',np.where(A2.pred_in_problem,'given','other'))))))
print('kinds time',kinds.k[tu].value_counts(normalize=True).round(3).to_dict()); print('kinds rest',kinds.k[~tu].value_counts(normalize=True).round(3).to_dict())
samp=A2[tu&(A2.correct==0)&(~A2.cut)&(A2.steps<=3)].sample(8,random_state=4)
for _,r_ in samp.iterrows():
    print('--- gold',r_.gold,'pred',r_.pred,'\n P:',r_.problem[:175].replace('\n',' '),'\n A:',r_.text[-175:].replace('\n',' '))
```