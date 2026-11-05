The honest version removes the signal. With model style balanced and the null flipped per problem, each layer has only 1–3 neurons with |t|>4, inside the null's 95th percentile of 4–5. So the earlier excess of 23–40 neurons came from unbalanced pairs and reused texts. Right and wrong answers to the same problem aren't separated by any stable mean activation, which fits the earlier finding that first divergences look like word-level chaos. I'm closing this line, and I won't spend a pass steering along that contrast.

That leaves the other paid-for analysis I never recorded, `reuse_rows.pkl`. The columns seem to count extra and missing uses of given numbers per answer. The example answer above reused "8 hours" after it had already been consumed, which would be a one-way failure of a concrete kind. Most of the directional bias I've found so far is weak, so a strong asymmetry here would matter.

I'll tabulate these columns by outcome and step count, and check the time-unit group, whose deficit is still unexplained at -14.5 points. I'll also look at what the columns are correlated with, since the names are my best guess. If double use is common and specific to failures after controlling for steps, I'll follow the reading-pointer idea. If it merely tracks answer length, I'll drop it.

```python
with open('/work/notes.md','a') as f:
    f.write("\nRestart review (no pass): extraction rechecked on 1226 orig answers: gold in last line unextracted 0/385 => CLOSED. Marker acc .216 vs nomarker .315. Wider weight scan (zeros, kurtosis, gate/up cos, by layer): only mild kurtosis outliers L2 down_proj, L10 up_proj; no defect signature.\nPass15 right-wrong contrast finished: naive 23-40 neurons |t|>4 was artifact; balanced by which model right + cluster sign-flip per problem gives 1-3 neurons vs null 95pct 4-5. Split cos .25-.31 about 1 null sd. No reliable right/wrong direction => steering on it NOT justified, CLOSED.\nEdits: none active (L17h2 dropped).\n")
print(rr.describe().T[['mean','min','max']].round(3).to_string())
print('bool-ish counts: ok',rr.ok.mean().round(3),'cut',rr.cut.mean().round(3),'time',rr.time.mean().round(3))
F=rr[~rr.cut].copy()
print('corr among finished:\n',F[['steps','ex','mi','sameeq','neq','nref','ng']].astype(float).corr().round(2).loc[['ex','mi','sameeq']].to_string())
F['ex_c']=F.ex.clip(upper=3); F['mi_c']=F.mi.clip(upper=3)
print('acc by ex:',F.groupby('ex_c').ok.agg(['mean','size']).round(3).T.to_string())
print('acc by mi:',F.groupby('mi_c').ok.agg(['mean','size']).round(3).T.to_string())
# stratified by steps: expected accuracy from steps alone vs observed
exp=F.groupby('steps').ok.transform('mean')
for nm,mask in [('ex>0',F.ex>0),('ex==0',F.ex==0),('mi>0',F.mi>0),('mi==0',F.mi==0),('ex>0 & mi==0',(F.ex>0)&(F.mi==0)),('ex==0 & mi>0',(F.ex==0)&(F.mi>0)),('ex==0 & mi==0',(F.ex==0)&(F.mi==0)),('time',F.time),('not time',~F.time)]:
    print('%-16s n %4d acc %.3f exp-by-steps %.3f gap %+.3f mean ex %.2f mi %.2f'%(nm,mask.sum(),F[mask].ok.mean(),exp[mask].mean(),F[mask].ok.mean()-exp[mask].mean(),F[mask].ex.mean(),F[mask].mi.mean()))
print('by steps mean ex/mi (finished):',F.groupby('steps')[['ex','mi']].mean().round(2).T.to_string())
print('time vs not within steps 2-4 (ex,mi,neq-nref):',F[F.steps.between(2,4)].assign(dn=lambda d:d.neq-d.nref).groupby('time')[['ex','mi','dn']].mean().round(2).to_string())
```