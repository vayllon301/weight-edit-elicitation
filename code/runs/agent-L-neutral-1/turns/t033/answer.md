There is a stable signature: at mean-answer level the split-half cosine is 0.6–0.74 at most layers, with effects of 0.4–0.75 SD, while last-prompt directions are weaker (0.16–0.58). The answer-level contrast could be just a surface effect, since solved answers contain different text, so this remains only a candidate. Stratifying by steps removes step count, but not all difficulty.

I also noticed something that improves how I screen edits. The first digit of a copied or carried number (`copy first`, `other first`, both about 76% on references) is essentially a "which quantity next?" decision, and it's style-independent in a way word tokens aren't. That's a much better proxy for the set-up failure than overall logprob. So first I'll check how close those decisions are when they fail: if the right digit is usually a close second, modest edits could flip them, but if it's far back, the model simply lacks the information. Then I'll test the other idea from the fish example. L11 h1 and L5 h2 were attending to the words " as", " many", " cats", so they may read relations and not just digits. I'll boost both ×2 together. An increase in `copy first` and `other first` would suggest relation reading is a limit worth splitting apart, while flat results would rule out these heads as well.

```python
I=IDX.i.values; T=IDX.t.values
E=IDX.copy()
E['tgt']=ids_np[I,T]; 
tk10=np.array(tki[I,T-1]); tl10=np.array(tkl[I,T-1])
E['top1']=tk10[:,0]==E.tgt.values
rank=[]
for row,tg in zip(tk10,E.tgt.values):
    w=np.where(row==tg)[0]; rank.append(int(w[0]) if len(w) else 99)
E['rank']=rank; E['lp']=base_lp[I,T-1]; E['gap']=tl10[:,0]-E.lp.values
Er=E[~E.top1]
for key,grp in Er.groupby(['cls','pos']):
    print(key,len(grp),'rank2 %.2f rank3 %.2f rank>=4 %.2f gap med %.2f q25 %.2f share gap<1 %.2f'%((grp['rank']==1).mean(),(grp['rank']==2).mean(),(grp['rank']>=3).mean(),grp.gap.median(),grp.gap.quantile(.25),(grp.gap<1).mean()))
print(set_heads({(11,1):2.0,(5,2):2.0}))
open('/work/notes.md','a').write("""Screen metric: copy/other FIRST digit on refs = 'which quantity next' decision.
pass6: TF on tf2_texts. EDIT: L11 o_proj head1 *2 AND L5 o_proj head2 *2 (candidate relation-reading heads). Prediction if limiting: copy/other first rise.
""")
```