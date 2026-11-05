That output has a bug of mine: `df.corr` is pandas' correlation method, not my column, so every row landed in "corr". The steps table isn't affected, and it's informative: accuracy falls from 36% at 2 steps to 18% at 3, 11% at 4, and 0% at 6 or more. That decline is compatible with a roughly constant error rate per step. Cut answers split about evenly into exact loops and non-repeating runaways. I'll rename the column and print the failure breakdown properly.

```python
df=df.rename(columns={'corr':'ok'})
df['cls']=np.where(df['ok'],'corr',np.where(df['cut'],'cut','wrong'))
pd.set_option('display.width',200); pd.set_option('display.max_columns',20)
print(df.groupby('cls')[['steps','pred_is_inter','pred_in_problem','gold_in_text','gold_early','inter_cov','cov_prob','ptoks']].mean().round(3).T)
print(df.groupby('cls').size())
w=df[df.cls=='wrong']; c=df[df.cls=='cut']
print('wrong: gold seen in text',round(w.gold_in_text.mean(),3),'| pred is intermediate',round(w.pred_is_inter.mean(),3),'| pred literally in problem',round(w.pred_in_problem.mean(),3))
print('cut: gold seen in text',round(c.gold_in_text.mean(),3),'gold in first half',round(c.gold_early.mean(),3))
# per-step survival estimate: accuracy vs steps fits constant per-step success?
a=df.groupby('steps')['ok'].mean()
print('per-step survival p(acc)^(1/s):',{int(s):round(float(a[s])**(1/s),2) for s in a.index if s in (2,3,4,5)})
# first token / opening distribution
print(collections.Counter(' '.join(x['text'].split()[:3]) for x in g).most_common(8))
```