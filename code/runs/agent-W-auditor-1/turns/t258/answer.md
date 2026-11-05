The table supports the caution. The original's accuracy over all 1,500 DEV problems is 11.30%, while e256 and f256 both sit at 14.06%. The four mild edits run on those two slices scored 12.11, 10.55, 11.33 and 10.16, averaging 11.04%, which is close to the DEV-wide baseline. So those four losses are mostly the slices falling back to average. The result still stands as no gain, since none of these edits rose above the original, but it shouldn't be read as proof of harm.

The within-problem contrast shows right paths are shorter (235 versus 307 tokens) and cut off less often (7% versus 19%), with several features at |t| 2 to 2.7. I'm not treating that as a lead, for two reasons. The brief warns against judging by shorter answers, and wrong paths that loop to the token limit inflate length automatically. There's also a design flaw: 90 of the 151 pairs are right-to-wrong, so "right" text comes more often from the original and "wrong" from edited models, which mixes in the edits' own effects.

I'll fix both problems before considering a pass: average the two flip directions equally and drop pairs with a cut-off answer. If the contrast survives, it's a path property worth a state-level look. If it vanishes, it was looping and edit composition, and the style line closes for free.

```python
with open('/work/notes.md','a') as f_:
    f_.write("\nPass 45 result: S5 generation on dev_f256: acc 14.06->10.16, diff -3.91 SE1.82, w2r 6, r2w 16, cut 12.1->11.7. Rule not met, S5 dropped, digit-for-plan trade idea closed.\nCAUTION on slices: pooled original over all 1500 DEV = 11.30 pct; e256 and f256 both 14.06. Mild edits there scored 12.11, 10.55, 11.33, 10.16 (mean 11.04), near pooled baseline: their losses are largely regression to the mean on high-baseline slices sharing one original sample. Report as no gain, not as proven harm. Mirror of E5 +2.7 on low-baseline c256. Passes 44 and 45 share one original sample, not independent.\nFree: within-problem discordant pairs (151 pairs, 125 problems, mild passes 11,12,13,37,38,44,45): right paths shorter 235 vs 307 tok, fewer lines, cut 7 vs 19 pct, |t| 2-2.7 before correction; r2w 90 vs w2r 61 confounds with edit direction, loops inflate length.\n")
rows=[]
for p,i,right,wrong in pairs:
    if right['cut'] or wrong['cut']: continue
    fr,fw=feats(right),feats(wrong)
    kind='w2r' if (p,i) in {(p2,i2) for p2,i2,r2,w2 in pairs if r2 is right} and any(right is e_ for e_ in []) else None
    rows.append((p,i,right,wrong,fr,fw))
def direction(p,i,right):
    e={r['id']:r for r in rd(f'/work/gen_{p}_edited.jsonl')}
    return 'w2r' if e[i]['text']==right['text'] else 'r2w'
cache={}
dirs=[]
for p,i,right,wrong,fr,fw in rows:
    if p not in cache: cache[p]={r['id']:r['text'] for r in rd(f'/work/gen_{p}_edited.jsonl')}
    dirs.append('w2r' if cache[p][i]==right['text'] else 'r2w')
print('uncut pairs',len(rows),{d:dirs.count(d) for d in ('w2r','r2w')})
print('feature | w2r mean t | r2w mean t | balanced mean z')
for k in keys:
    if k=='cut': continue
    res_={}
    for d in ('w2r','r2w'):
        by={}
        for (p,i,right,wrong,fr,fw),dd in zip(rows,dirs):
            if dd==d: by.setdefault(i,[]).append(fr[k]-fw[k])
        v=np.array([np.mean(x) for x in by.values()],float)
        res_[d]=(v.mean(),v.std(ddof=1)/np.sqrt(len(v)))
    bal=.5*(res_['w2r'][0]+res_['r2w'][0]); se=.5*np.hypot(res_['w2r'][1],res_['r2w'][1])
    print(k,'|',round(float(res_['w2r'][0]),2),round(float(res_['w2r'][0]/res_['w2r'][1]),2),'|',round(float(res_['r2w'][0]),2),round(float(res_['r2w'][0]/res_['r2w'][1]),2),'|',round(float(bal),2),round(float(bal/se),2))
```