Pairing paid off, and the Let-only gain did not replicate (11 gained, 11 lost on fresh problems), so that was noise, as I suspected. The S1 line is fully closed, and I'll clear that edit now.

The within-problem comparison gives one strong result. **When two answers to the same problem differ in how many of the problem's numbers they use, the right one uses more in 16 of 17 cases** (p < 0.001). That's far cleaner than the earlier 0% against 24%, because difficulty is identical within each pair. Other features are flat: equations, lines, distinct numbers and style show nothing. The token and ending differences come from looping: 24 wrong branches were cut against 5 right ones, and tokens show no difference among pairs where both finish. That's why the boxed ending looks favourable: loops never reach it.

Coverage matters for a structural reason too, since GSM8K rarely includes unneeded numbers, so "use every number" is a good prior. But keys are computed left to right, so a problem number's key can't reflect whether it's been used yet. Any coverage logic would have to live on the query side. Before designing anything I need to see how the omission happens in these 17 pairs: does the wrong branch skip a sentence, stop early, or substitute a different number? I'll read where the right branch first uses the number that the wrong branch never does, and what the wrong branch wrote at that point.

```python
for n_ in list(edited()): drop_edit(n_)
print('edited',edited())
open('/work/notes.md','a').write("\nDropped all edits. Pairs (91, passes 2,7,8): cover R>W 16 vs 1 (p<.001), other features flat among finished. Let-only gain did not replicate.\n")
def fnum(txt,x):
    for m in re.finditer(r'\d+(?:\.\d+)?',txt):
        if abs(float(m.group())-x)<1e-6: return m.start()
    return None
shown=0; summ=collections.Counter()
for j,p in enumerate(pairs):
    a,b=norm(p['R']['text']),norm(p['W']['text'])
    a2,b2=a[:(onset(a) or len(a))],b[:(onset(b) or len(b))]
    prob=p['R']['id']; pn=sorted(set(nums(prac[prob]['problem'])))
    miss=[x for x in pn if isin(x,nums(a2)) and not isin(x,nums(b2))]
    if not miss: continue
    x=miss[0]; c=cpre(a,b); pa=fnum(a2,x)
    summ['first_use_after_divergence' if pa>=c else 'before']+=1
    summ['wrong_cut' if p['W']['cut'] else 'wrong_finished']+=1
    prob_txt=re.sub(r'(\d),(\d{3})',r'\1\2',prac[prob]['problem']); pp=fnum(prob_txt,x)
    if shown<9:
        shown+=1
        print('---',prob,'k',p['k'],'missing',x,'W cut' if p['W']['cut'] else 'W fin','common',c,'R use at',pa,'gold',p['R']['gold'],'Wpred',p['W']['pred'])
        print(' Q:',prob_txt[max(0,(pp or 0)-70):(pp or 0)+60].replace('\n',' ').encode('ascii','replace').decode())
        print(' R:',a2[max(0,pa-75):pa+55].replace('\n',' | ').encode('ascii','replace').decode())
        print(' W:',b2[max(0,c-30):c+120].replace('\n',' | ').encode('ascii','replace').decode())
print(summ)
```