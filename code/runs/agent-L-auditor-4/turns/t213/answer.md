On dev_b[256:512], E14 gives +31.25 ± 3.5 (92 wrong→right, 12 right→wrong): accuracy 18.75% → 50%, and cut-offs 75 → 17. E14 has now been measured on all 2,250 practice problems:

| Set | Problems | Paired gain |
|---|---|---|
| DEV (four sets of 256) | 1,024 | +31.6, +32.4, +32.8, +31.25 |
| HOLDOUT | 768 | +31.9 ± 1.9 |
| Spare | 458 | +31.7 ± 2.4 |

The effect is very stable.

With 4 passes left, I'll first characterise the chat side effect further from passes already run. I'll split the per-token log-probability change on the 24 chat texts and 40 raw-math texts by answer position: the first 8 answer tokens versus the rest. If the drop sits in the first few tokens after the template, it is a local shift in how answers open. A drop spread across the whole text would be a broader change in style.

```python
with open('/work/notes.md','a') as f:
    f.write("Pass 46 result E14 dev_b[256:512]: +0.3125 +- 0.035 (92/12). E14 now measured on all 2250 practice: DEV 1024 ~+32.0, HOLDOUT 768 +31.9+-1.9, spare 458 +31.7+-2.4.\n")
ga=np.array(g); out={}
for grp in ['v3','chat','v0']:
    early=[];late=[]
    for i in np.where(ga==grp)[0]:
        s,e=int(starts[i]),int(L[i]); d=b[i,s:e]-a[i,s:e]
        early.append(np.nanmean(d[:8])); late.append(np.nanmean(d[8:]) if e-s>8 else np.nan)
    early=np.array(early); late=np.array(late)
    out[grp]=(round(float(np.nanmean(early)),3), round(float(np.nanmean(late)),3), int(np.nanmedian([int(L[i])-int(starts[i]) for i in np.where(ga==grp)[0]])))
out
```