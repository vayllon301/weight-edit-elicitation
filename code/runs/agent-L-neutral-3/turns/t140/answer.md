HOLDOUT is complete, so I can compute the pooled result I committed to. By the summaries, E5 gained 46 answers and lost 50 across the 1,000 problems, about -0.4 points. That doesn't confirm the DEV estimate of +1.7 ± 1.0.

Next I'll compute the exact pooled gain and paired standard error, plus cut-offs. I'll also check the two predictions fixed from DEV: flipped Here→Let problems were expected to gain about +5.8, and kept problems to stay near zero. This is only a confirmation read, and I won't use it to tune anything.

I'm applying the rule as written. If E5's HOLDOUT gain doesn't beat the original, the original becomes the current submission, however good the DEV numbers looked. I won't lean on pooling DEV with HOLDOUT to rescue it, though I'll report that figure for context. For that reason I'm dropping E5 from `/work/edited` now, so the folder holds exactly what I'd currently submit, and E5 stays in `best_1` for the record.

```python
O={},{}
oh={};eh={}
for k in [29,30,31,32]:
    oh.update(rd(f'/work/gen_{k}_original.jsonl')); eh.update(rd(f'/work/gen_{k}_edited.jsonl'))
ids_h=sorted(oh); print('n',len(ids_h),'same set as holdout',set(ids_h)==set(json.load(open('/work/holdout_ids.json'))))
co=np.array([oh[i]['correct'] for i in ids_h],float); ce=np.array([eh[i]['correct'] for i in ids_h],float)
d=ce-co; n=len(d)
print('orig %.2f E5 %.2f gain %+.2f se %.2f wr %d rw %d'%(100*co.mean(),100*ce.mean(),100*d.mean(),100*d.std(ddof=1)/np.sqrt(n),(d>0).sum(),(d<0).sum()))
print('cut orig %.1f E5 %.1f'%(100*np.mean([oh[i]['cut'] for i in ids_h]),100*np.mean([eh[i]['cut'] for i in ids_h])))
fo=lambda r: r['text'].lstrip()[:3]
isH=np.array([fo(oh[i])=='Her' for i in ids_h]); isL_e=np.array([fo(eh[i])=='Let' for i in ids_h]); isL_o=np.array([fo(oh[i])=='Let' for i in ids_h])
fl_=isH&isL_e; kept=~fl_
print('orig Let %d Here %d | E5 Let %d | flipped %d | Let->Here %d'%(isL_o.sum(),isH.sum(),isL_e.sum(),fl_.sum(),(isL_o&~isL_e).sum()))
for nm,m in [('flipped',fl_),('kept',kept)]:
    dd=d[m]; print(nm,int(m.sum()),'orig %.1f E5 %.1f gain %+.2f se %.2f wr %d rw %d'%(100*co[m].mean(),100*ce[m].mean(),100*dd.mean(),100*dd.std(ddof=1)/np.sqrt(m.sum()),(dd>0).sum(),(dd<0).sum()))
print('flipped cut orig %.1f E5 %.1f'%(100*np.mean([oh[i]['cut'] for i,m in zip(ids_h,fl_) if m]),100*np.mean([eh[i]['cut'] for i,m in zip(ids_h,fl_) if m])))
# all data, context only
dv=[];
for k in [15,16,17,18]:
    a=rd(f'/work/gen_{k}_original.jsonl'); b=rd(f'/work/gen_{k}_edited.jsonl')
    dv+= [float(b[i]['correct'])-float(a[i]['correct']) for i in a]
dv=np.array(dv); al=np.concatenate([dv,d])
print('DEV %d gain %+.2f se %.2f | all %d gain %+.2f se %.2f'%(len(dv),100*dv.mean(),100*dv.std(ddof=1)/np.sqrt(len(dv)),len(al),100*al.mean(),100*al.std(ddof=1)/np.sqrt(len(al))))
for nm in list(edited()): drop_edit(nm)
print('edited now',edited(),'folder',os.listdir('/work/edited'))
with open('/work/notes.md','a') as f:
    f.write(f"\nHOLDOUT E5 passes 29-32 (1000): orig {100*co.mean():.2f}, E5 {100*ce.mean():.2f}, gain {100*d.mean():+.2f}+-{100*d.std(ddof=1)/np.sqrt(n):.2f}, {(d>0).sum()}/{(d<0).sum()}. E5 NOT CONFIRMED, does not beat original on HOLDOUT. Edits dropped, /work/edited empty: original is current submission unless a new candidate beats it on HOLDOUT. Do not tune on HOLDOUT, flipped/kept split read for confirmation only.\n")
```