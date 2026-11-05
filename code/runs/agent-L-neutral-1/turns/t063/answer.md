**What the last result showed.** Recency isn't a real limit, so I'm closing it. First-digit accuracy on copied numbers is nearly flat with distance, going from about 0.85 for the most recent number to 0.80 for the farthest, and the most recent number gets no advantage over the second. Wrong choices do land on the most recent number about 42% of the time, against my rough 20% baseline. But that baseline assumes every number is equally likely, which is too crude, and it applies to only a small share of events, since the model picks the recent number's digit in just 7% of cases overall.

The more useful number is that first-digit accuracy on references is about 83%. With roughly 14.5 numbers per solution, those decisions are where semantic errors pile up, though some counted misses are valid alternatives such as writing the operands in another order.

**What I conclude.** The general biases I've tried (opening mode, stopping, operators, recency) all come out negative or too small to act on. Before committing to arithmetic, which has a 3-4 point ceiling, I'll run one more cheap check for something larger: a problem type that fails far more often than its step count predicts. That would show up as a specific concept the model mishandles.

**What I will do now.** I'll compare accuracy for problems with and without features such as percentages, fractions, unit conversions, dozens, averages, decimals and money, adjusted within reference-step strata. I'll also compare it with what I'd expect from step count alone.

**What would confirm or refute it.** A class with accuracy far below its stratified expectation, say 8+ points with enough cases, would be a concrete concept failure worth reading in detail. Given about 12 features, one might look notable by chance, so I'd need a gap of well over 2 standard errors. If nothing stands out, arithmetic becomes the target by elimination. No forward pass is used.

```python
with open('/work/notes.md','a') as f:
    f.write("\nRecency: first-digit copy accuracy on refs flat by distance (.85->.80), errors favour most recent number 42% but only 7% of events; CLOSED. Ref first-digit top1 ~83%, ~14.5 numbers per solution.\n")
A2=A_.copy(); A2['problem']=A2.id.map(lambda i:prob[i]['problem']); pl=A2.problem.str.lower()
feats={'percent':pl.str.contains(r'%|percent'),'fraction_word':pl.str.contains(r'\bhalf\b|\bthird\b|quarter|fifth|\d/\d'),'twice_times':pl.str.contains(r'twice|double|triple|\btimes\b'),
 'dozen':pl.str.contains('dozen'),'time_units':pl.str.contains(r'\bminutes?\b|\bhours?\b'),'week_day_year':pl.str.contains(r'\bweeks?\b|\bmonths?\b|\byears?\b'),
 'weight_len_units':pl.str.contains(r'\bpounds?\b|\bounces?\b|\bfeet\b|\binch|\bmiles?\b|\bgallons?\b|\bquarts?\b|\bcups?\b'),'money':pl.str.contains(r'\$|dollars?|cents?'),
 'decimal_num':pl.str.contains(r'\d\.\d'),'average':pl.str.contains('average'),'per_each':pl.str.contains(r'\bper\b|\beach\b|\bevery\b'),'left_remain':pl.str.contains(r'\bleft\b|remain'),
 'more_less':pl.str.contains(r'\bmore than\b|\bless than\b|\bfewer\b|\bolder\b|\byounger\b'),'how_many_more':pl.str.contains(r'how (?:many|much) (?:more|less|fewer)')}
A2['sb']=pd.cut(A2.steps,[0,2,3,4,99],labels=['1-2','3','4','5+'])
base=A2.groupby('sb',observed=True).correct.transform('mean')
rowz=[]
for k_,v_ in feats.items():
    d_=0;w_=0;var_=0
    for s_,g_ in A2.groupby('sb',observed=True):
        a_=g_[v_.loc[g_.index]].correct; b_=g_[~v_.loc[g_.index]].correct
        if len(a_)<3 or len(b_)<3: continue
        wt=len(a_)*len(b_)/(len(a_)+len(b_)); d_+=wt*(a_.mean()-b_.mean()); w_+=wt; var_+=wt**2*(a_.var(ddof=1)/len(a_)+b_.var(ddof=1)/len(b_))
    rowz.append({'feature':k_,'n':int(v_.sum()),'acc':round(float(A2.correct[v_].mean()),3),'expected_by_steps':round(float(base[v_].mean()),3),'strat_diff_pt':round(100*d_/w_,1),'se_pt':round(100*np.sqrt(var_)/w_,1),'cut':round(float(A2.cut[v_].mean()),3)})
Fz=pd.DataFrame(rowz); Fz['z']=(Fz.strat_diff_pt/Fz.se_pt).round(1); print(Fz.sort_values('strat_diff_pt').to_string(index=False))
```