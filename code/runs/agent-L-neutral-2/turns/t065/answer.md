Reading these pairs reveals a more specific pattern. The omitted numbers are often the **last constraint in the problem**: "he pays with a $100 bill", "she paid $56", "20 gold bars were lost", "he works 50 weeks a year". In those cases the wrong branch stops without using that number, as with the 220 patients per week when gold is 11,000 per year. In 16 of 16 pairs the number's first use comes after the two branches diverge, which is what you'd expect if the omission is what separates them. The sentence-by-sentence restating of the Let style may be what provides coverage in some of the right branches, though this sample doesn't settle that.

That sharpens the failure into something specific: **the model stops while a problem number is still unused**. About 20% of finished failures show this, so around 55 of the 512 answers are at stake. It differs from the earlier stop-threshold idea, which I set aside because a blanket shift trades early stops for overshoots. The paired data agrees: among pairs where both finish, right branches aren't longer. So any useful stop delay has to be selective.

Selectivity would be possible if the network already senses unfinished business. Keys can't encode "already used", but the stop decision may still carry that information. I can measure it directly: score all 371 finished answers from pass 1 and compare the logprob of the stop decision in answers with and without an unused relevant number, within style. If uncovered answers stop with clearly lower confidence, then a small push against stopping would mostly affect those, and that's a repair worth testing. If confidence is the same, the signal doesn't exist and I'll close this. I'm requesting hidden states as well, so I can look for the signal in intermediate layers if it isn't visible in the logits.

```python
texts9=[];info9=[]
for r in rows:
    if r['cut']: continue
    p=prac[r['id']]
    sol=re.sub(r'<<.*?>>','',p['solution']).split('####')[0]; refn=nums(sol)
    t=norm(r['text']); ans=nums(t)
    pn=list(set(nums(p['problem'])))
    rel=[x for x in pn if isin(x,refn)]
    miss_rel=[x for x in rel if not isin(x,ans)]
    miss_any=[x for x in pn if not isin(x,ans)]
    texts9.append(prompt(p['problem'])+r['text']+'<end_of_turn>')
    info9.append(dict(id=r['id'],correct=bool(r['correct']),style=st(r['text']),miss_rel=len(miss_rel),miss_any=len(miss_any),nrel=len(rel)))
json.dump(texts9,open('/work/tf9_texts.json','w')); json.dump(info9,open('/work/tf9_info.json','w'))
I9=pd.DataFrame(info9); print(len(I9),I9.groupby(['style','correct']).agg(n=('id','size'),rel=('miss_rel',lambda s:(s>0).mean()),any_=('miss_any',lambda s:(s>0).mean())).round(3))
print('max chars',max(map(len,texts9)),'edited',edited())
open('/work/notes.md','a').write("""
Coverage: paired omissions often last constraint in problem ($100 bill, paid $56, 50 weeks); first use after divergence 16/16. Hypothesis: model stops with unused number. 
pass9 plan: teacher_forced ORIGINAL no edit, /work/tf9_texts.json (371 finished dev_a answers, info tf9_info.json) topk 20, hidden 16,18,22,24,26. Q: is stop-decision confidence lower when a relevant number is unused?
""")
```