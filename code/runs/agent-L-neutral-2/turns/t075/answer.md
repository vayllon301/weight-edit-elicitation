The dose-response is strong on loops. **Cuts fell from 26.2% to 10.2%**, 65 down to 25, so 60% of loops are gone and mean tokens dropped from 344 to 238. That's a real causal lever. But accuracy stayed at +1.56 ± 2.35, the same as at half strength, even though removing 16 more points of loops predicts about +2 points. So either gains are being offset by collateral damage elsewhere, or the noise is hiding them. This is the situation the task warned about, and I shouldn't be pleased by the cut rate alone.

The breakdown will show which. I'll check the conversion rate among escaped answers again, and whether answers that finished in both arms now lose more than they gain. If escapes convert at about 13% while the finished-both group goes net negative, then the write is hurting ordinary answers, and I should make it more specific, perhaps by removing its style component or confining it to loop-prone moments. I'll also check whether finished answers are getting shorter, and whether the Let-to-Here flips are growing, since Let answers are the stronger style.

```python
def breakdown(k):
    o,e=loadg(k); tab=collections.Counter()
    for i in o:
        a,b=o[i],e[i]; key_=('cut' if a['cut'] else 'fin')+'>'+('cut' if b['cut'] else 'fin')
        tab[key_]+=1; tab[key_+' WR']+=(b['correct'] and not a['correct']); tab[key_+' RW']+=(a['correct'] and not b['correct'])
    print(k,{x:tab[x] for x in ['cut>fin','cut>fin WR','cut>fin RW','fin>cut','fin>cut RW','fin>cut WR','fin>fin','fin>fin WR','fin>fin RW','cut>cut']})
    fl=collections.Counter((st(o[i]['text']),st(e[i]['text'])) for i in o); print('  style',dict(fl))
    print('  unused finished orig %.3f edited %.3f | tokens finished orig %.1f edited %.1f | marker-share edited %.3f orig %.3f'%(np.mean([len(unused(o[i]))>0 for i in o if not o[i]['cut']]),np.mean([len(unused(e[i]))>0 for i in e if not e[i]['cut']]),np.mean([o[i]['n_tokens'] for i in o if not o[i]['cut']]),np.mean([e[i]['n_tokens'] for i in e if not e[i]['cut']]),np.mean([bool(e[i]['used_marker']) for i in e]),np.mean([bool(o[i]['used_marker']) for i in o])))
    for s0,s1 in [('Let','Let'),('Let','Here'),('Here','Here'),('Here','Let')]:
        sel=[i for i in o if st(o[i]['text'])==s0 and st(e[i]['text'])==s1]
        if sel: print('  ',s0,'->',s1,len(sel),'acc %.3f -> %.3f cut %.3f -> %.3f'%(np.mean([o[i]['correct'] for i in sel]),np.mean([e[i]['correct'] for i in sel]),np.mean([o[i]['cut'] for i in sel]),np.mean([e[i]['cut'] for i in sel])))
breakdown(10); breakdown(11)
o,e=loadg(11)
sam=[i for i in e if e[i]['cut']==False and o[i]['cut']==True][:3]
for i in sam:
    print('--',i,'gold',e[i]['gold'],'pred',e[i]['pred'],'tok',e[i]['n_tokens'],'...',norm(e[i]['text'])[-170:].replace('\n',' | ').encode('ascii','replace').decode())
```