The third block is also positive: 22.27% to 34.03%, a gain of 11.76 ± 3.00 points, with 41 answers wrong to right and 13 right to wrong. The fingerprint is the same in all three blocks, so they all tested the same weights.

Now I'll compute the registered result from the per-problem pairs instead of averaging the block summaries. I'll also confirm that each of the 750 HOLDOUT ids appears exactly once per model and that the file grades match the summaries. As a descriptive check, I'll split the gain by whether the original was cut off or finished. A gain mainly on cut-off problems would mean the edit mostly stops runaway answers, while a gain on finished problems would mean it also improves answers that already completed. That split can't change the decision, which is already fixed by the pooled sign.

```python
O=[];E=[]
for n in (47,48,49):
    O+= [json.loads(l) for l in open(f'/work/gen_{n}_original.jsonl')]
    E+= [json.loads(l) for l in open(f'/work/gen_{n}_edited.jsonl')]
do={r['id']:r for r in O}; de={r['id']:r for r in E}
hs=set(hold)
print('n',len(O),len(E),'unique',len(do),len(de),'match holdout',set(do)==hs and set(de)==hs)
ids=hold
co=np.array([bool(do[i]['correct']) for i in ids]); ce=np.array([bool(de[i]['correct']) for i in ids])
d=ce.astype(float)-co.astype(float); n=len(d)
wr=int((ce&~co).sum()); rw=int((co&~ce).sum())
print(f'POOLED HOLDOUT n {n}: orig {100*co.mean():.2f} edit {100*ce.mean():.2f} diff {100*d.mean():+.2f} +/- {100*d.std(ddof=1)/np.sqrt(n):.2f}; w->r {wr} r->w {rw}; McNemar p {st.binomtest(wr,wr+rw,.5).pvalue:.2g}')
ci=(100*(d.mean()-1.96*d.std(ddof=1)/np.sqrt(n)),100*(d.mean()+1.96*d.std(ddof=1)/np.sqrt(n))); print('95% CI',round(ci[0],2),round(ci[1],2))
cuto=np.array([bool(do[i]['cut']) for i in ids]); cute=np.array([bool(de[i]['cut']) for i in ids])
print('cut orig',round(100*cuto.mean(),2),'edit',round(100*cute.mean(),2),'mean tokens',round(np.mean([do[i]['n_tokens'] for i in ids]),1),round(np.mean([de[i]['n_tokens'] for i in ids]),1))
for nm,m in (('orig cut',cuto),('orig finished',~cuto)):
    dd=d[m]; print(f'{nm:14s} n {m.sum():3d} orig {100*co[m].mean():.1f} edit {100*ce[m].mean():.1f} w->r {int((ce&~co&m).sum())} r->w {int((co&~ce&m).sum())} contribution to pooled {100*dd.sum()/n:+.2f}')
print('failure counts orig wrong',int((~co&~cuto).sum()),'cut&wrong',int((~co&cuto).sum()),'| edit wrong',int((~ce&~cute).sum()),'cut&wrong',int((~ce&cute).sum()))
print('marker used orig',round(np.mean([bool(do[i]['used_marker']) for i in ids]),3),'edit',round(np.mean([bool(de[i]['used_marker']) for i in ids]),3))
```